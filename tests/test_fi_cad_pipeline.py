"""FI-CAD 第一版建模流水线测试。

这些测试覆盖最关键的研究设计约束：
1. 2011 基线有心脏病者不能进入新发事件模型。
2. 后续结局只能来自随访/退出表。
3. train/valid/test 按 ID 切分后不能重叠。
4. 指标函数必须输出论文需要的假阳性率、假阴性率等字段。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.fi_cad.config import load_config
from src.fi_cad.data import build_baseline_features, build_outcome_table_from_frames, normalize_charls_id
from src.fi_cad.modeling import choose_threshold, compute_binary_metrics, make_pipeline, split_dataset, transformed_feature_names


class FiCadPipelineTests(unittest.TestCase):
    """验证建模流水线的核心行为。"""

    def test_baseline_heart_disease_is_excluded_from_incident_model(self) -> None:
        """2011 已有心脏病的人必须被排除。"""

        baseline = pd.DataFrame({"ID": ["p1", "p2", "p3"], "da007_7_": ["1", "2", "2"]})
        followup = {
            2013: pd.DataFrame({"ID": ["p1", "p2"], "zda007_7_": ["1", "1"]}),
            2015: pd.DataFrame({"ID": ["p3"], "zda007_7_": [pd.NA]}),
            2018: pd.DataFrame(),
            2020: pd.DataFrame(),
        }
        outcome = build_outcome_table_from_frames(baseline, followup, {})
        by_id = outcome.set_index("ID")

        self.assertFalse(bool(by_id.loc["p1", "include_in_modeling"]))
        self.assertTrue(bool(by_id.loc["p2", "include_in_modeling"]))
        self.assertEqual(int(by_id.loc["p2", "heart_related_event_by_2020"]), 1)

    def test_charls_wave1_id_is_normalized_to_followup_shape(self) -> None:
        """2011 的 11 位 ID 要能对齐后续 12 位 ID。"""

        ids = pd.Series(["09400411302", "094004113002"])
        normalized = normalize_charls_id(ids).tolist()

        self.assertEqual(normalized[0], "094004113002")
        self.assertEqual(normalized[0], normalized[1])

    def test_baseline_height_uses_qi002_not_qh006_arm_like_measure(self) -> None:
        """防止把 40 多厘米的体测列误命名为成人身高。

        这个测试对应一次真实建模异常：
        CatBoost 特征重要性里 `height_cm_2011` 排在第一，但追查发现上一版优先用了
        中位数约 43 的 `qh006`。这会把非身高体测变量伪装成身高，导致解释性报告严重误导。
        """

        demographic = pd.DataFrame(
            {
                "ID": ["09400411302"],
                "ba002_1": [1950],
                "rgender": [1],
                "bd001": [3],
                "be001": [1],
            }
        )
        health = pd.DataFrame(
            {
                "ID": ["09400411302"],
                "da001": [3],
                "da002": [3],
                "da059": [2],
                "da067": [1],
                "dc011": [1],
                "de006": [1],
                "da041": [2],
            }
        )
        biomarkers = pd.DataFrame(
            {
                "ID": ["09400411302"],
                "qh006": [43.0],
                "qi002": [168.0],
                "ql002": [70.0],
                "qa003": [130.0],
                "qa007": [128.0],
                "qm002": [88.0],
            }
        )

        features, _ = build_baseline_features(demographic, health, biomarkers, min_fi_observed_fraction=0.0)

        self.assertEqual(features.loc[0, "height_cm_2011"], 168.0)
        self.assertAlmostEqual(features.loc[0, "bmi_2011"], 70.0 / (1.68**2), places=6)

    def test_fi_uses_legacy_literature_eleven_item_definition(self) -> None:
        """FI 主列必须按旧论文/旧分支的 11 项等权口径计算。

        旧分支 `2011年FI+CVD及变量新.csv` 可还原出公式：
        10 个非 CVD 慢病缺陷 + BMI_to_FI，再除以 11。
        这里构造一个人：高血压=1、髋部骨折=1、BMI 肥胖=1，其余 8 项为 0，
        因此 FI 应为 3/11。
        """

        demographic = pd.DataFrame({"ID": ["09400411302"], "ba002_1": [1950], "rgender": [1]})
        health_values = {
            "ID": ["09400411302"],
            "da001": [3],
            "da002": [3],
            "dc011": [1],
            "de006": [1],
            "da041": [2],
            "da025": [1],
        }
        for number in [1, 3, 4, 5, 9, 10, 11, 13, 14]:
            health_values[f"da007_{number}_"] = [1 if number == 1 else 2]
        health = pd.DataFrame(health_values)
        biomarkers = pd.DataFrame({"ID": ["09400411302"], "qi002": [160.0], "ql002": [80.0]})

        features, _ = build_baseline_features(demographic, health, biomarkers, min_fi_observed_fraction=0.0)

        self.assertAlmostEqual(features.loc[0, "bmi_to_fi_2011"], 1.0)
        self.assertAlmostEqual(features.loc[0, "fi_2011"], 3.0 / 11.0, places=6)

    def test_split_dataset_has_no_id_overlap(self) -> None:
        """同一个 ID 不能同时出现在训练、验证和测试中。"""

        config = load_config(None)
        dataset = pd.DataFrame(
            {
                "ID": [f"p{i}" for i in range(80)],
                "feature_a": np.arange(80),
                "feature_b": np.arange(80) % 3,
                "heart_related_event_by_2020": [0, 1] * 40,
            }
        )
        split = split_dataset(dataset, "heart_related_event_by_2020", config)
        groups = split.split_table.groupby("split")["ID"].apply(set).to_dict()

        self.assertTrue(groups["train"].isdisjoint(groups["valid"]))
        self.assertTrue(groups["train"].isdisjoint(groups["test"]))
        self.assertTrue(groups["valid"].isdisjoint(groups["test"]))

    def test_metrics_include_false_positive_and_false_negative_rates(self) -> None:
        """指标表必须包含 FPR/FNR 等论文诊断字段。"""

        y_true = pd.Series([0, 0, 1, 1])
        y_score = np.array([0.1, 0.7, 0.6, 0.9])
        threshold, info = choose_threshold(y_true, y_score, "balanced_youden_f1")
        metrics = compute_binary_metrics(y_true, y_score, threshold)

        self.assertIn("fpr", metrics)
        self.assertIn("fnr", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("selected_threshold", info)

    def test_categorical_codes_are_one_hot_inside_pipeline(self) -> None:
        """分类编码列必须在训练 pipeline 内展开为哑变量。

        这样可以避免把性别、教育、婚姻、自评健康等编码当成连续数值。
        测试重点不是模型效果，而是确认预处理器 fit 后的特征名里出现 One-Hot 展开列。
        """

        x_train = pd.DataFrame(
            {
                "age_2011": [60, 62, 64, 66],
                "sex_code_2011": [1, 2, 1, 2],
                "education_code_2011": [1, 2, 3, 2],
                "fi_2011": [0.1, 0.2, 0.3, 0.4],
            }
        )
        y_train = pd.Series([0, 1, 0, 1])
        pipeline = make_pipeline("logistic_regression", y_train, {}, list(x_train.columns), 42)
        pipeline.fit(x_train, y_train)
        names = transformed_feature_names(pipeline, list(x_train.columns))

        self.assertIn("age_2011", names)
        self.assertIn("fi_2011", names)
        self.assertTrue(any(name.startswith("sex_code_2011_") for name in names))
        self.assertTrue(any(name.startswith("education_code_2011_") for name in names))


if __name__ == "__main__":
    unittest.main()
