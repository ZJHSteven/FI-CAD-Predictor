"""CHARLS 纵向建模数据集构建。

本文件把已经整理好的 `data/curated` 转成第一版论文建模表。
核心原则：
1. 预测变量只来自 2011 基线，避免把未来信息泄露给模型。
2. 结局只来自 2013/2015/2018/2020 随访和退出问卷。
3. 2011 已经有心脏病相关诊断的人必须排除，因为本任务预测的是“新发事件”。
4. FI 先实现为可复核的缺陷比例：慢病、功能困难、抑郁/疼痛/自评健康等缺陷项的均值。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_CODES = {"", " ", "d", "r", "m", "nan", "none", "<na>"}


@dataclass(frozen=True)
class DatasetBuildResult:
    """数据集构建结果。

    字段说明：
    - dataset: 可以直接进入模型训练的宽表。
    - outcome_table: 人级结局表，记录基线排除、随访观察和首次事件年份。
    - baseline_features: 只含 2011 基线特征的宽表。
    - variable_dictionary: 输出变量解释，方便论文和后续人工审查。
    - missingness: 每个特征的缺失率。
    - high_correlation_pairs: 高相关特征对，用于排查共线性和潜在重复变量。
    """

    dataset: pd.DataFrame
    outcome_table: pd.DataFrame
    baseline_features: pd.DataFrame
    variable_dictionary: pd.DataFrame
    missingness: pd.DataFrame
    high_correlation_pairs: pd.DataFrame


def read_parquet_table(curated_root: Path, relative_path: str) -> pd.DataFrame:
    """读取 curated 层 Parquet 表。

    输入：
    - curated_root: `data/curated` 根目录。
    - relative_path: 相对 curated_root 的表路径。

    输出：
    - pandas DataFrame。
    """

    path = curated_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"建模所需数据表不存在：{path}")
    frame = pd.read_parquet(path)
    if "ID" in frame.columns:
        frame["ID"] = normalize_charls_id(frame["ID"])
    return frame


def normalize_charls_id(series: pd.Series) -> pd.Series:
    """统一 CHARLS 跨波次个人 ID。

    背景：
    - 2011 Wave1 的个人 ID 在当前数据里常见为 11 位，例如 `09400411302`。
    - 2013 之后同一个人常见为 12 位，例如 `094004113002`。
    - 这不是不同人，而是中间补了 1 位成员序号占位。

    核心逻辑：
    - 先把 ID 当字符串保留前导 0。
    - 11 位 ID 规范化为“前 9 位 + 0 + 后 2 位”。
    - 12 位 ID 原样保留。
    - 其他长度不强行猜测，只去除空格后返回，便于后续暴露问题。
    """

    text = series.astype("string").str.strip()
    return text.mask(text.str.len() == 11, text.str.slice(0, 9) + "0" + text.str.slice(9))


def load_metadata(curated_root: Path, relative_path: str) -> dict[str, Any]:
    """读取 curated 层 metadata JSON。"""

    path = curated_root / relative_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_numeric(series: pd.Series) -> pd.Series:
    """把 CHARLS 原始编码列安全转成数值列。

    CHARLS 的 Stata 特殊缺失和拒答/不知道常以 `d`、`r` 等字符串保存。
    这里统一转成 NaN，避免模型把这些文字当成正常类别。
    """

    cleaned = series.astype("string").str.strip().str.lower()
    cleaned = cleaned.mask(cleaned.isin(MISSING_CODES))
    return pd.to_numeric(cleaned, errors="coerce")


def plausible_numeric_range(series: pd.Series, *, minimum: float, maximum: float) -> pd.Series:
    """把数值列转成数值，并把明显不合理的测量值置为缺失。

    输入：
    - series: CHARLS 原始列。
    - minimum/maximum: 允许保留的闭区间范围。

    输出：
    - 超出范围的值会变成 NaN。

    为什么需要这个函数：
    - 体测列里常见 `993/999` 这类特殊编码，不能当成真实身高、体重或血压。
    - 如果不先清理，树模型很容易把这些异常码当成强信号，最后得到很荒谬的特征重要性。
    """

    numeric = to_numeric(series)
    return numeric.where(numeric.between(minimum, maximum, inclusive="both"))


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    """把 `1=Yes, 2=No` 的 CHARLS 列转成 1/0。

    输出：
    - Yes -> 1.0
    - No -> 0.0
    - 其他缺失/拒答/不知道 -> NaN
    """

    numeric = to_numeric(series)
    return pd.Series(np.select([numeric == 1, numeric == 2], [1.0, 0.0], default=np.nan), index=series.index)


def difficulty_to_deficit(series: pd.Series) -> pd.Series:
    """把功能困难题转成 FI 缺陷项。

    CHARLS 这类题通常是：
    - 1: 没有困难
    - 2: 有困难但能做
    - 3: 有困难且需要帮助
    - 4: 不能做

    因此 2/3/4 记为缺陷，1 记为无缺陷。
    """

    numeric = to_numeric(series)
    return pd.Series(np.select([numeric == 1, numeric >= 2], [0.0, 1.0], default=np.nan), index=series.index)


def bmi_to_literature_fi_deficit(bmi: pd.Series) -> pd.Series:
    """把 BMI 转成旧论文/旧分支使用的 FI 缺陷项。

    输入：
    - bmi: 已经清洗过的 BMI 数值。

    输出：
    - 正常 BMI: 0.0
    - 超重: 0.5
    - 肥胖或偏瘦: 1.0

    口径来源：
    - `origin/main:data/raw/2011年FI+CVD及变量新.csv` 里有 `BMI_to_FI` 列。
    - 对照旧数据可还原其大致分档：BMI 24~28 记 0.5，BMI >=28 记 1，BMI <18.5 也按营养不足风险记 1。
    - 这个变量是旧 FI 公式的第 11 个缺陷项。
    """

    return pd.Series(
        np.select(
            [
                bmi < 18.5,
                (bmi >= 18.5) & (bmi < 24),
                (bmi >= 24) & (bmi < 28),
                bmi >= 28,
            ],
            [1.0, 0.0, 0.5, 1.0],
            default=np.nan,
        ),
        index=bmi.index,
    )


def bad_health_to_deficit(series: pd.Series, *, threshold: float) -> pd.Series:
    """把自评健康、抑郁程度、疼痛程度等有序题转成缺陷项。

    输入：
    - threshold: 大于等于该数值时视作存在健康缺陷。
    """

    numeric = to_numeric(series)
    return pd.Series(np.select([numeric < threshold, numeric >= threshold], [0.0, 1.0], default=np.nan), index=series.index)


def wave_event_from_frame(frame: pd.DataFrame, event_columns: list[str], *, one_means_yes_only: bool = False) -> pd.Series:
    """从一个随访表中提取“是否发生心脏病相关事件”。

    输入：
    - frame: 某一波健康表或退出问卷。
    - event_columns: 可代表心脏病相关事件的列名列表。
    - one_means_yes_only: True 表示只有 `1` 代表事件，其他值不强行解释为明确 No。

    输出：
    - 以 ID 为索引的 0/1 序列。同一个 ID 多行时，只要任一行阳性即阳性。
    """

    if frame.empty or "ID" not in frame.columns:
        return pd.Series(dtype="float64")
    available = [column for column in event_columns if column in frame.columns]
    if not available:
        return pd.Series(0.0, index=pd.Index(frame["ID"].astype("string"), name="ID"))

    signal = pd.DataFrame(index=frame.index)
    for column in available:
        numeric = to_numeric(frame[column])
        if one_means_yes_only:
            signal[column] = (numeric == 1).astype(float)
        else:
            signal[column] = pd.Series(np.select([numeric == 1, numeric == 2], [1.0, 0.0], default=np.nan), index=frame.index)
    event = signal.max(axis=1, skipna=True).fillna(0.0)
    event_by_id = event.groupby(normalize_charls_id(frame["ID"])).max()
    event_by_id.index.name = "ID"
    return event_by_id


def build_outcome_table_from_frames(
    baseline_health: pd.DataFrame,
    followup_health: dict[int, pd.DataFrame],
    exit_modules: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """构建人级结局表。

    输入：
    - baseline_health: 2011 健康状况与功能表。
    - followup_health: 后续波次健康表，键为年份。
    - exit_modules: 退出/死亡相关表，键为年份。

    输出：
    - 每个 2011 基线受访者一行，包含基线是否有病、是否可纳入、随访是否观察到、是否新发、首次事件年份。
    """

    if "ID" not in baseline_health.columns or "da007_7_" not in baseline_health.columns:
        raise ValueError("2011 健康表必须包含 ID 和 da007_7_，才能定义基线心脏病状态。")

    baseline = baseline_health[["ID", "da007_7_"]].copy()
    baseline["ID"] = normalize_charls_id(baseline["ID"])
    baseline["baseline_heart_disease"] = yes_no_to_binary(baseline["da007_7_"])
    outcome = baseline[["ID", "baseline_heart_disease"]].drop_duplicates("ID").set_index("ID")
    outcome["eligible_baseline_no_heart_disease"] = outcome["baseline_heart_disease"].eq(0.0)

    wave_columns = {
        2013: ["zda007_7_", "da007_7_", "da007_w2_5"],
        2015: ["zda007_7_", "da007_7_", "da007_w2_5"],
        2018: ["da007_7_", "zdisease_7_", "da007_w2_5"],
        2020: ["zdisease_7_", "da002_7_"],
    }
    exit_columns = {
        2013: ["zda007_7_", "exda007_w2_2_7_", "exda007_w2_5"],
        2020: ["exda003", "xedisease_3_", "xezdisease_3_"],
    }

    observed_any = pd.Series(False, index=outcome.index)
    first_event_year = pd.Series(np.nan, index=outcome.index, dtype="float64")

    for year in [2013, 2015, 2018, 2020]:
        health_frame = followup_health.get(year, pd.DataFrame())
        health_event = wave_event_from_frame(
            health_frame,
            wave_columns.get(year, []),
            one_means_yes_only=year in {2018, 2020},
        )
        exit_frame = exit_modules.get(year, pd.DataFrame())
        exit_event = wave_event_from_frame(exit_frame, exit_columns.get(year, []), one_means_yes_only=True)
        combined = pd.concat([health_event, exit_event], axis=1).max(axis=1, skipna=True).fillna(0.0)
        combined = combined.reindex(outcome.index).fillna(0.0)

        observed_ids = set()
        if not health_frame.empty and "ID" in health_frame.columns:
            observed_ids.update(normalize_charls_id(health_frame["ID"]).dropna().tolist())
        if not exit_frame.empty and "ID" in exit_frame.columns:
            observed_ids.update(normalize_charls_id(exit_frame["ID"]).dropna().tolist())
        observed_wave = outcome.index.isin(observed_ids)
        observed_any = observed_any | observed_wave
        event_wave = combined.eq(1.0) & outcome["eligible_baseline_no_heart_disease"]
        first_event_year = first_event_year.mask(first_event_year.isna() & event_wave, float(year))
        outcome[f"heart_event_{year}"] = combined.astype(float)
        outcome[f"observed_{year}"] = observed_wave

    outcome["observed_followup_any"] = observed_any
    outcome["first_heart_event_year"] = first_event_year
    outcome["include_in_modeling"] = outcome["eligible_baseline_no_heart_disease"] & outcome["observed_followup_any"]
    for horizon in [2013, 2015, 2018, 2020]:
        observed_columns = [f"observed_{year}" for year in [2013, 2015, 2018, 2020] if year <= horizon]
        event_columns = [f"heart_event_{year}" for year in [2013, 2015, 2018, 2020] if year <= horizon]
        observed_by_horizon = outcome[observed_columns].any(axis=1)
        event_by_horizon = outcome[event_columns].eq(1.0).any(axis=1) & outcome["eligible_baseline_no_heart_disease"]
        target = event_by_horizon.astype(float).where(
            outcome["eligible_baseline_no_heart_disease"] & observed_by_horizon,
            np.nan,
        )
        outcome[f"observed_followup_by_{horizon}"] = observed_by_horizon
        outcome[f"heart_related_event_by_{horizon}"] = target
    return outcome.reset_index()


def build_baseline_features(
    demographic: pd.DataFrame,
    baseline_health: pd.DataFrame,
    biomarkers: pd.DataFrame,
    blood: pd.DataFrame | None = None,
    *,
    include_blood: bool = False,
    min_fi_observed_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建 2011 基线特征表。

    输入：
    - demographic: 2011 基本信息表。
    - baseline_health: 2011 健康状况与功能表。
    - biomarkers: 2011 体检信息表。
    - blood: 2011 血检表，可选；第一版主模型默认不用。
    - include_blood: 是否纳入血检增强特征。
    - min_fi_observed_fraction: FI 至少需要多少缺陷项非缺失。

    输出：
    - baseline_features: 人级基线特征表。
    - variable_dictionary: 变量解释表。
    """

    base = baseline_health[["ID"]].drop_duplicates().copy()
    base["ID"] = normalize_charls_id(base["ID"])
    dictionary_rows: list[dict[str, str]] = []

    def add_feature(name: str, values: pd.Series, label: str, group: str) -> None:
        base[name] = values.reindex(base.index).to_numpy()
        dictionary_rows.append({"feature": name, "group": group, "label": label})

    demo = demographic.drop_duplicates("ID").copy()
    demo["ID"] = normalize_charls_id(demo["ID"])
    demo = demo.set_index("ID").reindex(base["ID"]).reset_index()
    health = baseline_health.drop_duplicates("ID").copy()
    health["ID"] = normalize_charls_id(health["ID"])
    health = health.set_index("ID").reindex(base["ID"]).reset_index()
    bio = biomarkers.drop_duplicates("ID").copy()
    bio["ID"] = normalize_charls_id(bio["ID"])
    bio = bio.set_index("ID").reindex(base["ID"]).reset_index()

    add_feature("age_2011", 2011 - to_numeric(demo.get("ba002_1", pd.Series(index=demo.index))), "2011 年年龄", "demographic")
    add_feature("sex_code_2011", to_numeric(demo.get("rgender", pd.Series(index=demo.index))), "性别原始编码", "demographic")
    add_feature("education_code_2011", to_numeric(demo.get("bd001", pd.Series(index=demo.index))), "最高受教育程度原始编码", "demographic")
    add_feature("marital_code_2011", to_numeric(demo.get("be001", pd.Series(index=demo.index))), "婚姻状态原始编码", "demographic")

    add_feature("self_rated_health_2011", to_numeric(health.get("da001", pd.Series(index=health.index))), "自评健康原始编码", "health")
    add_feature("health_compared_peer_2011", to_numeric(health.get("da002", pd.Series(index=health.index))), "与同龄人相比健康原始编码", "health")
    add_feature("current_smoking_2011", yes_no_to_binary(health.get("da059", pd.Series(index=health.index))), "当前是否吸烟", "lifestyle")
    add_feature("alcohol_frequency_code_2011", to_numeric(health.get("da067", pd.Series(index=health.index))), "过去一年饮酒频率原始编码", "lifestyle")

    chronic_deficits: list[pd.Series] = []
    chronic_by_number: dict[int, pd.Series] = {}
    for number in range(1, 15):
        if number == 7:
            continue
        column = f"da007_{number}_"
        if column not in health.columns:
            continue
        feature = f"chronic_disease_{number}_2011"
        values = yes_no_to_binary(health[column])
        add_feature(feature, values, f"2011 慢病诊断条目 {number}，心脏病条目 7 已排除", "chronic")
        chronic_deficits.append(values)
        chronic_by_number[number] = values

    function_deficits: list[pd.Series] = []
    for number in range(1, 21):
        column = f"db{number:03d}"
        if column not in health.columns:
            continue
        feature = f"functional_difficulty_{number}_2011"
        values = difficulty_to_deficit(health[column])
        add_feature(feature, values, f"2011 功能困难缺陷项 {column}", "function")
        function_deficits.append(values)

    extra_deficits = [
        bad_health_to_deficit(health.get("da001", pd.Series(index=health.index)), threshold=4),
        bad_health_to_deficit(health.get("da002", pd.Series(index=health.index)), threshold=4),
        bad_health_to_deficit(health.get("dc011", pd.Series(index=health.index)), threshold=3),
        bad_health_to_deficit(health.get("de006", pd.Series(index=health.index)), threshold=3),
        yes_no_to_binary(health.get("da041", pd.Series(index=health.index))),
    ]
    add_feature("depressed_frequency_2011", to_numeric(health.get("dc011", pd.Series(index=health.index))), "过去一周感到抑郁频率", "mental")
    add_feature("sad_or_depressed_2011", to_numeric(health.get("de006", pd.Series(index=health.index))), "悲伤、低落或抑郁程度", "mental")

    # 重要修正：
    # - 上一版错误地把 qh006 优先当作身高；真实数据中 qh006 中位数约 43，明显不是成人身高。
    # - qi002 的中位数约 158，更符合厘米制身高，所以这里改成只用 qi002 作为身高。
    # - 所有体测变量都先做合理范围过滤，避免 993/999 等特殊编码污染模型。
    height = plausible_numeric_range(bio.get("qi002", pd.Series(index=bio.index)), minimum=120, maximum=220)
    weight = plausible_numeric_range(bio.get("ql002", pd.Series(index=bio.index)), minimum=25, maximum=180)
    height_m = height / 100.0
    bmi = weight / (height_m**2)
    bmi = bmi.where(bmi.between(10, 60, inclusive="both"))
    bmi_fi_deficit = bmi_to_literature_fi_deficit(bmi)
    add_feature("height_cm_2011", height, "2011 体检测量身高 cm", "biomarker")
    add_feature("weight_kg_2011", weight, "2011 体检测量体重 kg", "biomarker")
    add_feature("bmi_2011", bmi, "2011 BMI，由身高体重计算", "biomarker")
    add_feature("bmi_to_fi_2011", bmi_fi_deficit, "旧论文 FI 的 BMI 缺陷项：正常=0、超重=0.5、肥胖或偏瘦=1", "fi_literature_component")

    systolic_1 = plausible_numeric_range(bio.get("qa003", pd.Series(index=bio.index)), minimum=70, maximum=260)
    systolic_2 = plausible_numeric_range(bio.get("qa007", pd.Series(index=bio.index)), minimum=70, maximum=260)
    systolic_mean = pd.concat([systolic_1, systolic_2], axis=1).mean(axis=1, skipna=True)
    waist = plausible_numeric_range(bio.get("qm002", pd.Series(index=bio.index)), minimum=45, maximum=180)
    add_feature("systolic_bp_mean_2011", systolic_mean, "2011 两次收缩压均值，异常码已过滤", "biomarker")
    add_feature("waist_cm_2011", waist, "2011 腰围/围度类体测 cm，异常码已过滤", "biomarker")

    literature_fi_items = {
        "hypertension": chronic_by_number.get(1, pd.Series(np.nan, index=health.index)),
        "chronic_lung_disease": chronic_by_number.get(5, pd.Series(np.nan, index=health.index)),
        "asthma": chronic_by_number.get(14, pd.Series(np.nan, index=health.index)),
        "stomach_or_digestive_disease": chronic_by_number.get(10, pd.Series(np.nan, index=health.index)),
        "arthritis_or_rheumatism": chronic_by_number.get(13, pd.Series(np.nan, index=health.index)),
        "fracture_hip": yes_no_to_binary(health.get("da025", pd.Series(index=health.index))),
        "psychiatric_disease": chronic_by_number.get(11, pd.Series(np.nan, index=health.index)),
        "diabetes": chronic_by_number.get(3, pd.Series(np.nan, index=health.index)),
        "cancer": chronic_by_number.get(4, pd.Series(np.nan, index=health.index)),
        "kidney_disease": chronic_by_number.get(9, pd.Series(np.nan, index=health.index)),
        "bmi_to_fi": bmi_fi_deficit,
    }
    literature_fi_matrix = pd.DataFrame(literature_fi_items, index=health.index)
    literature_observed_fraction = literature_fi_matrix.notna().mean(axis=1)
    literature_fi_score = literature_fi_matrix.mean(axis=1, skipna=True).where(literature_observed_fraction >= min_fi_observed_fraction)
    broad_fi_matrix = pd.concat(chronic_deficits + function_deficits + extra_deficits, axis=1)
    broad_observed_fraction = broad_fi_matrix.notna().mean(axis=1)
    broad_fi_score = broad_fi_matrix.mean(axis=1, skipna=True).where(broad_observed_fraction >= min_fi_observed_fraction)
    add_feature("fi_2011", literature_fi_score, "2011 虚弱指数：旧论文/旧分支 11 项等权 FI，心脏病和卒中不进入 FI", "fi")
    add_feature("fi_observed_fraction_2011", literature_observed_fraction, "旧论文 FI 11 个缺陷项的非缺失比例", "fi_quality")
    add_feature("fi_broad_exploratory_2011", broad_fi_score, "探索性宽口径 FI：慢病、功能困难和少量心理/自评健康项均值，仅作敏感性参考", "fi_exploratory")
    add_feature("fi_broad_observed_fraction_2011", broad_observed_fraction, "探索性宽口径 FI 缺陷项非缺失比例", "fi_quality")

    if include_blood and blood is not None and not blood.empty:
        blood_frame = blood.drop_duplicates("ID").copy()
        blood_frame["ID"] = normalize_charls_id(blood_frame["ID"])
        blood_frame = blood_frame.set_index("ID").reindex(base["ID"]).reset_index()
        for column, label in {
            "newglu": "血糖 mg/dl",
            "newcho": "总胆固醇 mg/dl",
            "newhdl": "HDL 胆固醇 mg/dl",
            "newldl": "LDL 胆固醇 mg/dl",
            "newcrp": "C 反应蛋白 mg/l",
            "newhba1c": "糖化血红蛋白 %",
            "qc1_vb004": "血红蛋白 g/dl",
        }.items():
            if column in blood_frame.columns:
                add_feature(f"{column}_2011", to_numeric(blood_frame[column]), f"2011 血检增强特征：{label}", "blood")

    variable_dictionary = pd.DataFrame(dictionary_rows)
    return base, variable_dictionary


def build_missingness_table(dataset: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """生成特征缺失率表。"""

    excluded = {"ID", target_column, "first_heart_event_year"}
    rows = []
    for column in dataset.columns:
        if column in excluded or column.startswith("heart_event_") or column.startswith("heart_related_event_by_") or column.startswith("observed_"):
            continue
        rows.append({"feature": column, "missing_rate": float(dataset[column].isna().mean())})
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def build_high_correlation_table(dataset: pd.DataFrame, target_column: str, threshold: float = 0.95) -> pd.DataFrame:
    """生成高相关变量对表。

    说明：
    - 这里只做训练前的排查报告，不在数据构建阶段直接删列。
    - 真正是否删列应由训练 pipeline 或研究者基于报告决定。
    """

    excluded = {"ID", target_column, "first_heart_event_year"}
    feature_frame = dataset[
        [
            column
            for column in dataset.columns
            if column not in excluded
            and not column.startswith("heart_event_")
            and not column.startswith("heart_related_event_by_")
            and not column.startswith("observed_")
        ]
    ]
    numeric = feature_frame.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(numeric_only=True).abs()
    rows = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "abs_correlation": float(value)})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_correlation"])


def build_modeling_dataset(config: dict[str, Any]) -> DatasetBuildResult:
    """按配置构建第一版建模数据集。"""

    curated_root = Path(config["data"]["curated_root"])
    include_blood = bool(config["dataset"].get("include_blood_enhanced_features", False))
    min_fi_fraction = float(config["dataset"].get("min_fi_observed_fraction", 0.20))
    target_column = str(config["dataset"].get("endpoint_name", "heart_related_event_by_2020"))

    baseline_health = read_parquet_table(curated_root, "2011-wave1/健康状况与功能/health_status_and_functioning.parquet")
    demographic = read_parquet_table(curated_root, "2011-wave1/基本信息/demographic_background.parquet")
    biomarkers = read_parquet_table(curated_root, "2011-wave1/体检信息/biomarkers.parquet")
    blood = read_parquet_table(curated_root, "2011-wave1/血检数据/Blood_20140429.parquet") if include_blood else None
    followup_health = {
        2013: read_parquet_table(curated_root, "2013-wave2/健康状况与功能/Health_Status_and_Functioning.parquet"),
        2015: read_parquet_table(curated_root, "2015-wave3/健康状况与功能/Health_Status_and_Functioning.parquet"),
        2018: read_parquet_table(curated_root, "2018-wave4/健康状况与功能/Health_Status_and_Functioning.parquet"),
        2020: read_parquet_table(curated_root, "2020-wave5/健康状况与功能/Health_Status_and_Functioning.parquet"),
    }
    exit_modules = {
        2013: read_parquet_table(curated_root, "2013-wave2/退出调查/Exit_Interview.parquet"),
        2020: read_parquet_table(curated_root, "2020-wave5/退出问卷/Exit_Module.parquet"),
    }

    outcome_table = build_outcome_table_from_frames(baseline_health, followup_health, exit_modules)
    baseline_features, variable_dictionary = build_baseline_features(
        demographic,
        baseline_health,
        biomarkers,
        blood,
        include_blood=include_blood,
        min_fi_observed_fraction=min_fi_fraction,
    )
    outcome_columns = [
        column
        for column in outcome_table.columns
        if column in {"ID", "baseline_heart_disease", "eligible_baseline_no_heart_disease", "observed_followup_any", "first_heart_event_year"}
        or column.startswith("observed_followup_by_")
        or column.startswith("heart_related_event_by_")
    ]
    if target_column not in outcome_columns:
        raise ValueError(f"配置中的终点列不存在：{target_column}")
    dataset = baseline_features.merge(
        outcome_table[outcome_columns],
        on="ID",
        how="inner",
    )
    dataset = dataset[dataset["eligible_baseline_no_heart_disease"] & dataset[target_column].notna()].copy()
    if dataset.empty:
        raise ValueError("建模数据集为空：请检查 CHARLS ID 规范化、基线排除条件和随访结局表。")
    dataset[target_column] = dataset[target_column].astype(int)
    missingness = build_missingness_table(dataset, target_column)
    high_corr = build_high_correlation_table(dataset, target_column)
    return DatasetBuildResult(dataset, outcome_table, baseline_features, variable_dictionary, missingness, high_corr)


def write_dataset_outputs(result: DatasetBuildResult, config: dict[str, Any]) -> dict[str, str]:
    """把数据集构建产物写入配置指定路径。"""

    paths = {
        "dataset": Path(config["data"]["dataset_path"]),
        "outcome_table": Path(config["data"]["outcome_table_path"]),
        "baseline_features": Path(config["data"]["baseline_feature_path"]),
        "variable_dictionary": Path(config["data"]["variable_dictionary_path"]),
        "missingness": Path(config["data"]["missingness_path"]),
        "high_correlation_pairs": Path(config["data"]["correlation_path"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_parquet(paths["dataset"], index=False)
    result.outcome_table.to_csv(paths["outcome_table"], index=False, encoding="utf-8-sig")
    result.baseline_features.to_csv(paths["baseline_features"], index=False, encoding="utf-8-sig")
    result.variable_dictionary.to_csv(paths["variable_dictionary"], index=False, encoding="utf-8-sig")
    result.missingness.to_csv(paths["missingness"], index=False, encoding="utf-8-sig")
    result.high_correlation_pairs.to_csv(paths["high_correlation_pairs"], index=False, encoding="utf-8-sig")
    return {key: str(path) for key, path in paths.items()}
