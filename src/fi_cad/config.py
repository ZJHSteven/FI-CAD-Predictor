"""配置读取工具。

本文件只负责一件事：把 YAML 配置读成普通字典，并提供少量安全默认值。
这样做的好处是：训练入口不需要关心配置文件格式，测试也可以直接传入字典。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "curated_root": "data/curated",
        "dataset_path": "output/datasets/modeling_dataset.parquet",
        "outcome_table_path": "output/tables/outcome_table.csv",
        "baseline_feature_path": "output/tables/baseline_features.csv",
        "variable_dictionary_path": "output/tables/variable_dictionary.csv",
        "missingness_path": "output/tables/feature_missingness.csv",
        "correlation_path": "output/tables/high_correlation_pairs.csv",
    },
    "run": {
        "output_root": "output/runs",
        "random_seed": 20260429,
        "test_size": 0.20,
        "valid_size": 0.20,
        "primary_metric": "roc_auc",
        "threshold_strategy": "balanced_youden_f1",
        "min_auc_warning": 0.70,
        "max_fpr_warning": 0.40,
        "min_recall_warning": 0.05,
        "all_negative_rate_warning": 0.98,
    },
    "dataset": {
        "endpoint_name": "heart_related_event_by_2020",
        "baseline_year": 2011,
        "horizon_year": 2020,
        "include_blood_enhanced_features": False,
        "min_fi_observed_fraction": 0.20,
    },
    "training": {
        "optuna_trials": 6,
        "optuna_timeout_seconds": 180,
        "models": ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个配置字典。

    输入：
    - base: 默认配置。
    - override: 用户配置。

    输出：
    - 合并后的新字典，不会原地修改 `base`。

    核心逻辑：
    - 如果两边同一个键都是字典，就继续向下合并。
    - 否则用 override 的值覆盖 base。
    """

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取建模配置。

    输入：
    - config_path: YAML 文件路径；为 None 时只返回默认配置。

    输出：
    - 合并默认值后的配置字典。
    """

    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件顶层必须是 YAML 字典：{path}")
    return deep_update(DEFAULT_CONFIG, loaded)
