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
from src.fi_cad.data import build_outcome_table_from_frames
from src.fi_cad.modeling import choose_threshold, compute_binary_metrics, split_dataset


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


if __name__ == "__main__":
    unittest.main()
