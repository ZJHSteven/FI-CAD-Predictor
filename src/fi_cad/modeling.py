"""模型训练、评估和图表输出工具。

本文件显式实现 train/valid/test 三段式流程：
1. 先按人 ID 切分。
2. 每个模型都把填补、标准化和分类器封装进 sklearn Pipeline。
3. 阈值只在验证集上选择，测试集只用于最终报告。
这样可以避免旧项目里“先全量预处理再切分”的数据泄露问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitData:
    """训练、验证、测试三份数据。"""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_valid: pd.DataFrame
    y_valid: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    split_table: pd.DataFrame


@dataclass(frozen=True)
class FeatureSet:
    """一次训练使用的特征集合。

    字段说明：
    - name: 特征集名称，会写入指标表和模型文件名。
    - description: 人类可读说明，用于报告解释。
    - features: 真正进入模型的列名列表。
    """

    name: str
    description: str
    features: list[str]


def git_commit() -> str:
    """返回当前 Git commit，用于写入 run_manifest。"""

    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def package_versions() -> dict[str, str]:
    """记录关键依赖版本，保证论文结果可追溯。"""

    versions: dict[str, str] = {}
    for name in [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "matplotlib",
        "shap",
        "tabulate",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def require_runtime_dependencies(config: dict[str, Any]) -> None:
    """检查训练必须依赖是否真的安装。

    这里故意不做“缺包就跳过”的降级处理：
    - SHAP 是解释性分析的核心产物，缺它就不应把训练 run 当作完整成功。
    - tabulate 虽然当前报告有手写表格兜底，但用户已经要求虚拟环境里补齐依赖，因此也纳入检查。
    """

    required = ["pandas", "numpy", "scikit-learn", "joblib", "matplotlib", "optuna", "tabulate"]
    if bool(config.get("interpretability", {}).get("require_shap", True)):
        required.append("shap")
    missing = []
    for package_name in required:
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package_name)
    if missing:
        install_command = f".venv\\Scripts\\python.exe -m pip install {' '.join(missing)}"
        raise RuntimeError(f"训练缺少关键依赖：{', '.join(missing)}。请先运行：{install_command}")


def feature_columns(dataset: pd.DataFrame, target_column: str) -> list[str]:
    """从建模宽表中提取可进入模型的特征列。"""

    blocked = {
        "ID",
        target_column,
        "baseline_heart_disease",
        "eligible_baseline_no_heart_disease",
        "observed_followup_any",
        "first_heart_event_year",
    }
    return [
        column
        for column in dataset.columns
        if column not in blocked
        and not column.startswith("heart_event_")
        and not column.startswith("observed_")
        # `fi_observed_fraction_2011` 这类列是“这个 FI 有多少项非缺失”的质量控制信息，
        # 不是生物医学特征。上一轮解释性输出显示它会被模型学到，因此训练阶段统一排除。
        and not column.endswith("observed_fraction_2011")
    ]


def configured_feature_sets(config: dict[str, Any], available_features: list[str]) -> list[FeatureSet]:
    """根据配置生成特征集。

    输入：
    - config: `configs/modeling.yaml` 中读出的配置。
    - available_features: 数据集中真实存在的候选特征。

    输出：
    - FeatureSet 列表。默认至少包含 `primary`。

    设计意图：
    - 主分析保留全部候选变量，回答“目前所有 2011 基线信息能预测到什么程度”。
    - 敏感性分析可以排除身高、体重、BMI、年龄、性别等粗变量，检查模型是否只是学到常识性体型/人口学信号。
    """

    feature_set_config = config.get("training", {}).get("feature_sets", {})
    if not feature_set_config:
        feature_set_config = {"primary": {"description": "主分析特征集。", "exclude": []}}
    feature_sets: list[FeatureSet] = []
    available = list(available_features)
    available_set = set(available)
    for name, item in feature_set_config.items():
        included = item.get("include")
        excluded = set(item.get("exclude", []))
        if included:
            included_set = set(included)
            missing_included = sorted(included_set - available_set)
            if missing_included:
                raise ValueError(f"特征集 {name} 指定纳入的列不存在：{missing_included}")
            base_features = [feature for feature in available if feature in included_set]
        else:
            base_features = available
        missing_excluded = sorted(excluded - available_set)
        if missing_excluded:
            raise ValueError(f"特征集 {name} 排除的列不存在：{missing_excluded}")
        features = [feature for feature in base_features if feature not in excluded]
        if not features:
            raise ValueError(f"特征集 {name} 没有任何可训练特征。")
        feature_sets.append(
            FeatureSet(
                name=str(name),
                description=str(item.get("description", "")),
                features=features,
            )
        )
    return feature_sets


def subset_split(split: SplitData, features: list[str]) -> SplitData:
    """在保持同一批 ID 切分不变的前提下，只替换特征列。

    这比为每个特征集重新随机切分更稳妥：
    - 不同特征集共享完全相同的 train/valid/test 人群。
    - 指标差异更能反映特征变化，而不是随机切分变化。
    """

    return SplitData(
        x_train=split.x_train[features],
        y_train=split.y_train,
        x_valid=split.x_valid[features],
        y_valid=split.y_valid,
        x_test=split.x_test[features],
        y_test=split.y_test,
        split_table=split.split_table,
    )


def categorical_feature_columns(feature_names: list[str]) -> list[str]:
    """列出需要按分类变量处理的基线编码列。

    这些列的数值大小本身没有连续测量含义：
    - 性别、婚姻、教育是类别编码。
    - 自评健康、同龄健康比较、饮酒频率虽然有顺序含义，但不同档位之间的距离不应默认相等。

    因此这里用 One-Hot，让模型学习每个档位的独立风险，而不是把 `1/2/3/4` 当作等距连续数。
    """

    configured = {
        "sex_code_2011",
        "education_code_2011",
        "marital_code_2011",
        "self_rated_health_2011",
        "health_compared_peer_2011",
        "alcohol_frequency_code_2011",
        "depressed_frequency_2011",
        "sad_or_depressed_2011",
    }
    return [feature for feature in feature_names if feature in configured]


def safe_stratify(y: pd.Series) -> pd.Series | None:
    """只有正负样本都足够时才启用 stratify。"""

    counts = y.value_counts()
    return y if len(counts) == 2 and counts.min() >= 2 else None


def split_dataset(dataset: pd.DataFrame, target_column: str, config: dict[str, Any]) -> SplitData:
    """按人 ID 切分 train/valid/test。

    输入：
    - dataset: build_dataset 产出的建模宽表。
    - target_column: 结局列名。
    - config: 建模配置。

    输出：
    - SplitData，包含三份 X/y 和每个 ID 的 split 标记。
    """

    seed = int(config["run"]["random_seed"])
    test_size = float(config["run"]["test_size"])
    valid_size = float(config["run"]["valid_size"])
    features = feature_columns(dataset, target_column)
    working = dataset[["ID", target_column, *features]].dropna(subset=[target_column]).copy()
    train_valid, test = train_test_split(
        working,
        test_size=test_size,
        random_state=seed,
        stratify=safe_stratify(working[target_column]),
    )
    relative_valid_size = valid_size / (1.0 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid_size,
        random_state=seed,
        stratify=safe_stratify(train_valid[target_column]),
    )

    split_table = pd.concat(
        [
            train[["ID"]].assign(split="train"),
            valid[["ID"]].assign(split="valid"),
            test[["ID"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    return SplitData(
        x_train=train[features],
        y_train=train[target_column].astype(int),
        x_valid=valid[features],
        y_valid=valid[target_column].astype(int),
        x_test=test[features],
        y_test=test[target_column].astype(int),
        split_table=split_table,
    )


def make_preprocessor(feature_names: list[str], *, scale: bool) -> ColumnTransformer:
    """构建只在训练集 fit 的预处理器。

    处理规则：
    - 连续变量：中位数填补；Logistic 回归额外标准化。
    - 分类编码变量：众数填补 + One-Hot，避免把类别编号误当作连续距离。
    - 所有 fit 都发生在训练集内部，避免数据泄露。
    """

    categorical_features = categorical_feature_columns(feature_names)
    numeric_features = [feature for feature in feature_names if feature not in categorical_features]
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(pipeline: Pipeline, fallback_features: list[str]) -> list[str]:
    """读取预处理后真正进入模型的特征名。

    One-Hot 后，一个原始列会展开成多个哑变量。
    特征重要性、SHAP 和报告必须使用展开后的列名，否则会出现“重要性数组长度”和“原始特征名长度”对不上。
    """

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        return list(fallback_features)


def class_ratio(y: pd.Series) -> float:
    """计算负/正样本比例，供不平衡模型权重使用。"""

    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return negatives / positives


def build_estimator(model_name: str, y_train: pd.Series, params: dict[str, Any], seed: int) -> Any:
    """按模型名创建分类器。"""

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=float(params.get("scale_pos_weight", class_ratio(y_train))),
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 30)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            scale_pos_weight=float(params.get("scale_pos_weight", class_ratio(y_train))),
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(params.get("iterations", 300)),
            depth=int(params.get("depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            auto_class_weights="Balanced",
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"未知模型：{model_name}")


def suggest_params(trial: optuna.Trial, model_name: str, ratio: float) -> dict[str, Any]:
    """给 Optuna 定义小而稳的搜索空间。"""

    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 0.03, 10.0, log=True)}
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 360),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 360),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 8.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", max(1.0, ratio * 0.5), max(1.1, ratio * 1.5)),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 360),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", max(1.0, ratio * 0.5), max(1.1, ratio * 1.5)),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 120, 360),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    raise ValueError(f"未知模型：{model_name}")


def predict_probability(pipeline: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    """统一获取阳性概率。"""

    probabilities = pipeline.predict_proba(x_frame)
    return probabilities[:, 1]


def choose_threshold(y_true: pd.Series, y_score: np.ndarray, strategy: str) -> tuple[float, dict[str, float]]:
    """在验证集上选择分类阈值。

    balanced_youden_f1 策略：
    - Youden 指数重视敏感度和特异度平衡。
    - F1 重视 precision/recall 折中。
    - 两者相加能避免只追 AUC 却在 0.5 阈值下全判负。
    """

    thresholds = np.unique(np.clip(y_score, 0.001, 0.999))
    if len(thresholds) == 0:
        return 0.5, {"youden_threshold": 0.5, "f1_threshold": 0.5}
    best_balanced = (0.5, -np.inf)
    best_f1 = (0.5, -np.inf)
    best_youden = (0.5, -np.inf)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        youden = sensitivity + specificity - 1.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced = youden + f1
        if f1 > best_f1[1]:
            best_f1 = (float(threshold), float(f1))
        if youden > best_youden[1]:
            best_youden = (float(threshold), float(youden))
        if balanced > best_balanced[1]:
            best_balanced = (float(threshold), float(balanced))
    selected = {"f1": best_f1[0], "youden": best_youden[0]}.get(strategy, best_balanced[0])
    return selected, {
        "selected_threshold": float(selected),
        "f1_threshold": best_f1[0],
        "f1_at_f1_threshold": best_f1[1],
        "youden_threshold": best_youden[0],
        "youden_at_youden_threshold": best_youden[1],
        "balanced_threshold": best_balanced[0],
        "balanced_score": best_balanced[1],
    }


def compute_binary_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """计算论文和诊断需要的二分类指标。"""

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.nunique() == 2 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "predicted_negative_rate": float(1.0 - np.mean(y_pred)),
    }


def make_pipeline(model_name: str, y_train: pd.Series, params: dict[str, Any], feature_names: list[str], seed: int) -> Pipeline:
    """创建完整 sklearn Pipeline。"""

    estimator = build_estimator(model_name, y_train, params, seed)
    preprocessor = make_preprocessor(feature_names, scale=model_name == "logistic_regression")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def tune_and_fit_model(model_name: str, split: SplitData, config: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    """用 Optuna 在验证集上调参，然后用最佳参数训练最终 pipeline。"""

    seed = int(config["run"]["random_seed"])
    trials = int(config["training"].get("optuna_trials", 6))
    timeout = int(config["training"].get("optuna_timeout_seconds", 180))
    feature_names = list(split.x_train.columns)
    ratio = class_ratio(split.y_train)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, ratio)
        pipeline = make_pipeline(model_name, split.y_train, params, feature_names, seed)
        pipeline.fit(split.x_train, split.y_train)
        score = predict_probability(pipeline, split.x_valid)
        return float(roc_auc_score(split.y_valid, score)) if split.y_valid.nunique() == 2 else 0.0

    if trials > 0:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
    else:
        best_params = {}
        best_value = np.nan
    pipeline = make_pipeline(model_name, split.y_train, best_params, feature_names, seed)
    pipeline.fit(split.x_train, split.y_train)
    return pipeline, {"best_params": best_params, "best_valid_roc_auc": best_value}


def model_diagnostic_status(metrics: dict[str, float], config: dict[str, Any]) -> tuple[str, str]:
    """根据测试集表现给模型打诊断状态。"""

    min_recall = float(config["run"].get("min_recall_warning", 0.05))
    min_auc = float(config["run"].get("min_auc_warning", 0.70))
    max_fpr = float(config["run"].get("max_fpr_warning", 0.40))
    max_negative = float(config["run"].get("all_negative_rate_warning", 0.98))
    if metrics["predicted_negative_rate"] >= max_negative:
        return "failed", "模型几乎全判负，需要检查阈值、类别不平衡和特征信号。"
    if metrics["recall"] <= min_recall:
        return "warning", "召回率过低，存在漏诊风险。"
    warnings = []
    if pd.notna(metrics["roc_auc"]) and metrics["roc_auc"] < min_auc:
        warnings.append(f"ROC-AUC 低于 {min_auc:.2f}，区分度偏弱")
    if metrics["fpr"] > max_fpr:
        warnings.append(f"FPR 高于 {max_fpr:.2f}，误报偏多")
    if warnings:
        return "warning", "；".join(warnings) + "。"
    return "ok", "通过基础诊断。"


def plot_dataset_diagnostics(dataset: pd.DataFrame, target_column: str, figure_dir: Path, table_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """输出数据集级别诊断图。

    图表说明：
    - FI 风险关系图：检查 FI 越高，事件率是否整体上升。
    - 相关性热图：只展示与结局绝对相关最高的一组变量，避免 50 多个特征挤在一起无法阅读。

    额外输出：
    - 每张图对应一份 CSV，放在 `tables/figure_data/`，便于不看图也能复核数值。
    """

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_data_dir = table_dir / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    data_paths: dict[str, str] = {}
    if "fi_2011" in dataset.columns:
        working = dataset[["fi_2011", target_column]].dropna().copy()
        if not working.empty and working["fi_2011"].nunique() >= 3:
            working["fi_bin"] = pd.qcut(working["fi_2011"], q=10, duplicates="drop")
            grouped = working.groupby("fi_bin", observed=True).agg(
                fi_mean=("fi_2011", "mean"),
                event_rate=(target_column, "mean"),
                n=(target_column, "size"),
            )
            plt.figure(figsize=(7, 5))
            plt.plot(grouped["fi_mean"], grouped["event_rate"], marker="o")
            for _, row in grouped.iterrows():
                plt.text(row["fi_mean"], row["event_rate"], str(int(row["n"])), fontsize=8)
            plt.xlabel("FI 2011 mean in bin")
            plt.ylabel("Observed event rate")
            plt.title("FI 2011 and incident heart-related event")
            path = figure_dir / "dataset_fi_risk_curve.png"
            plt.tight_layout()
            plt.savefig(path, dpi=180)
            plt.close()
            paths["fi_risk_curve"] = str(path)
            data_path = figure_data_dir / "dataset_fi_risk_curve_data.csv"
            grouped.reset_index(drop=True).to_csv(data_path, index=False, encoding="utf-8-sig")
            data_paths["fi_risk_curve_data"] = str(data_path)

    features = feature_columns(dataset, target_column)
    numeric = dataset[[*features, target_column]].apply(pd.to_numeric, errors="coerce")
    corr_to_target = numeric.corr(numeric_only=True)[target_column].drop(labels=[target_column], errors="ignore").abs()
    target_corr_path = figure_data_dir / "dataset_target_correlations.csv"
    corr_to_target.sort_values(ascending=False).rename_axis("feature").reset_index(name="abs_correlation_to_target").to_csv(
        target_corr_path,
        index=False,
        encoding="utf-8-sig",
    )
    data_paths["target_correlations"] = str(target_corr_path)
    top_features = corr_to_target.sort_values(ascending=False).head(20).index.tolist()
    if len(top_features) >= 2:
        corr = numeric[top_features].corr(numeric_only=True)
        plt.figure(figsize=(9, 8))
        image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(image, fraction=0.046, pad=0.04)
        plt.xticks(range(len(top_features)), top_features, rotation=90, fontsize=7)
        plt.yticks(range(len(top_features)), top_features, fontsize=7)
        plt.title("Top feature correlation heatmap")
        path = figure_dir / "dataset_correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths["correlation_heatmap"] = str(path)
        corr_path = figure_data_dir / "dataset_correlation_heatmap_data.csv"
        corr.to_csv(corr_path, encoding="utf-8-sig")
        data_paths["correlation_heatmap_data"] = str(corr_path)
    return paths, data_paths


def plot_curves(
    model_id: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    figure_dir: Path,
    table_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """输出 ROC、PR、校准和混淆矩阵图，并保存底层数据。"""

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_data_dir = table_dir / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    data_paths: dict[str, str] = {}
    if y_true.nunique() == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_true, y_score):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        path = figure_dir / f"{model_id}_roc.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths["roc"] = str(path)
        roc_data_path = figure_data_dir / f"{model_id}_roc_data.csv"
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(roc_data_path, index=False, encoding="utf-8-sig")
        data_paths["roc_data"] = str(roc_data_path)

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_score):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        path = figure_dir / f"{model_id}_pr.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths["pr"] = str(path)
        pr_data_path = figure_data_dir / f"{model_id}_pr_data.csv"
        pr_threshold_column = np.append(pr_thresholds, np.nan)
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_threshold_column}).to_csv(
            pr_data_path,
            index=False,
            encoding="utf-8-sig",
        )
        data_paths["pr_data"] = str(pr_data_path)

        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=8, strategy="quantile")
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed event rate")
        path = figure_dir / f"{model_id}_calibration.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths["calibration"] = str(path)
        calibration_data_path = figure_data_dir / f"{model_id}_calibration_data.csv"
        pd.DataFrame({"predicted_probability_mean": prob_pred, "observed_event_rate": prob_true}).to_csv(
            calibration_data_path,
            index=False,
            encoding="utf-8-sig",
        )
        data_paths["calibration_data"] = str(calibration_data_path)

    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        plt.text(column, row, str(value), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    path = figure_dir / f"{model_id}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["confusion_matrix"] = str(path)
    confusion_data_path = figure_data_dir / f"{model_id}_confusion_matrix_data.csv"
    pd.DataFrame(
        [
            {"true_label": 0, "predicted_label": 0, "count": int(matrix[0, 0])},
            {"true_label": 0, "predicted_label": 1, "count": int(matrix[0, 1])},
            {"true_label": 1, "predicted_label": 0, "count": int(matrix[1, 0])},
            {"true_label": 1, "predicted_label": 1, "count": int(matrix[1, 1])},
        ]
    ).to_csv(confusion_data_path, index=False, encoding="utf-8-sig")
    data_paths["confusion_matrix_data"] = str(confusion_data_path)
    score_data_path = figure_data_dir / f"{model_id}_test_scores.csv"
    pd.DataFrame({"y_true": y_true.to_numpy(), "y_score": y_score, "y_pred": y_pred}).to_csv(
        score_data_path,
        index=False,
        encoding="utf-8-sig",
    )
    data_paths["test_scores"] = str(score_data_path)
    return paths, data_paths


def plot_feature_importance(
    model_id: str,
    pipeline: Pipeline,
    feature_names: list[str],
    figure_dir: Path,
    table_dir: Path,
) -> tuple[str | None, str | None, pd.DataFrame]:
    """输出模型特征重要性图和 CSV。

    Logistic 使用系数绝对值；树模型使用 feature_importances_。
    """

    model = pipeline.named_steps["model"]
    values: np.ndarray | None = None
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    if values is None or len(values) != len(feature_names):
        return None, None, pd.DataFrame()
    summary = pd.DataFrame({"feature": feature_names, "importance": values})
    summary = summary.sort_values("importance", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    explainability_dir = table_dir / "explainability"
    explainability_dir.mkdir(parents=True, exist_ok=True)
    table_path = explainability_dir / f"{model_id}_feature_importance.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8-sig")
    order = np.argsort(values)[-20:]
    plt.figure(figsize=(8, 7))
    plt.barh(np.asarray(feature_names)[order], values[order])
    plt.xlabel("Importance")
    path = figure_dir / f"{model_id}_feature_importance.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path), str(table_path), summary


def train_all_models(config: dict[str, Any]) -> dict[str, Any]:
    """训练配置中列出的所有模型并写出完整 run 产物。"""

    require_runtime_dependencies(config)
    target_column = str(config["dataset"]["endpoint_name"])
    dataset_path = Path(config["data"]["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"建模数据集不存在，请先运行 python -m src.fi_cad.build_dataset：{dataset_path}")
    dataset = pd.read_parquet(dataset_path)
    split = split_dataset(dataset, target_column, config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config["run"]["output_root"]) / timestamp
    table_dir = run_dir / "tables"
    model_dir = run_dir / "models"
    figure_dir = run_dir / "figures"
    report_dir = run_dir / "reports"
    for directory in [table_dir, model_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    split.split_table.to_csv(table_dir / "splits.csv", index=False, encoding="utf-8-sig")
    dataset_figures, dataset_figure_data = plot_dataset_diagnostics(dataset, target_column, figure_dir, table_dir)

    metrics_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    model_outputs: dict[str, Any] = {}
    feature_sets = configured_feature_sets(config, list(split.x_train.columns))
    feature_set_rows = [
        {
            "feature_set": feature_set.name,
            "description": feature_set.description,
            "feature_count": len(feature_set.features),
            "features": ";".join(feature_set.features),
        }
        for feature_set in feature_sets
    ]
    feature_sets_path = table_dir / "feature_sets.csv"
    pd.DataFrame(feature_set_rows).to_csv(feature_sets_path, index=False, encoding="utf-8-sig")

    for feature_set in feature_sets:
        active_split = subset_split(split, feature_set.features)
        feature_names = list(active_split.x_train.columns)
        for model_name in config["training"]["models"]:
            model_id = f"{feature_set.name}__{model_name}"
            try:
                pipeline, tuning_info = tune_and_fit_model(model_name, active_split, config)
                valid_score = predict_probability(pipeline, active_split.x_valid)
                threshold, threshold_info = choose_threshold(
                    active_split.y_valid,
                    valid_score,
                    str(config["run"]["threshold_strategy"]),
                )
                test_score = predict_probability(pipeline, active_split.x_test)
                metrics = compute_binary_metrics(active_split.y_test, test_score, threshold)
                status, message = model_diagnostic_status(metrics, config)
                model_path = model_dir / f"{model_id}.joblib"
                joblib.dump(pipeline, model_path)
                figure_paths, figure_data_paths = plot_curves(model_id, active_split.y_test, test_score, threshold, figure_dir, table_dir)
                model_feature_names = transformed_feature_names(pipeline, feature_names)
                importance_path, importance_table_path, importance_table = plot_feature_importance(
                    model_id,
                    pipeline,
                    model_feature_names,
                    figure_dir,
                    table_dir,
                )
                if importance_path:
                    figure_paths["feature_importance"] = importance_path
                metrics_rows.append(
                    {
                        "feature_set": feature_set.name,
                        "model": model_name,
                        "status": status,
                        "message": message,
                        "threshold": threshold,
                        **metrics,
                    }
                )
                threshold_rows.append({"feature_set": feature_set.name, "model": model_name, **threshold_info, **tuning_info})
                model_outputs[model_id] = {
                    "feature_set": feature_set.name,
                    "model": model_name,
                    "model_path": str(model_path),
                    "figures": figure_paths,
                    "figure_data": figure_data_paths,
                    "feature_importance_table": importance_table_path,
                    "transformed_feature_count": len(model_feature_names),
                    "top_features": importance_table.head(10).to_dict(orient="records") if not importance_table.empty else [],
                    "status": status,
                    "message": message,
                    "best_params": tuning_info["best_params"],
                }
            except Exception as exc:  # noqa: BLE001
                metrics_rows.append(
                    {
                        "feature_set": feature_set.name,
                        "model": model_name,
                        "status": "failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                model_outputs[model_id] = {
                    "feature_set": feature_set.name,
                    "model": model_name,
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }

    metrics = pd.DataFrame(metrics_rows)
    thresholds = pd.DataFrame(threshold_rows)
    metrics_path = table_dir / "model_metrics.csv"
    thresholds_path = table_dir / "thresholds.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")

    manifest = {
        "run_dir": str(run_dir),
        "created_at": timestamp,
        "git_commit": git_commit(),
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "sample_count": int(len(dataset)),
        "positive_rate": float(dataset[target_column].mean()),
        "feature_count": len(feature_columns(dataset, target_column)),
        "tables": {
            "metrics": str(metrics_path),
            "thresholds": str(thresholds_path),
            "splits": str(table_dir / "splits.csv"),
            "feature_sets": str(feature_sets_path),
        },
        "figures": dataset_figures,
        "figure_data": dataset_figure_data,
        "models": model_outputs,
        "package_versions": package_versions(),
        "config": config,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = Path(config["run"]["output_root"]) / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")
    return manifest
