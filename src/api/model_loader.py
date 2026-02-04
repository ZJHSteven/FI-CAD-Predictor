# -*- coding: utf-8 -*-
# 本文件负责模型与配置的加载：
# 1) 读取API配置（阈值、路径、默认值策略等）
# 2) 读取训练数据用于计算默认值与特征列
# 3) 读取模型指标并筛选可用模型
# 4) 加载最终参与集成的模型

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os

import pandas as pd
import numpy as np
import joblib

from src.utils.config_loader import create_config_loader
from src.api.errors import ApiError


@dataclass
class ModelBundle:
    """
    统一打包模型加载结果，便于预测器直接使用。

    Attributes:
        models: 模型对象字典（key为模型名称）
        weights: 模型权重字典（key为模型名称）
        feature_columns: 训练特征列名列表
        defaults: 各特征的默认值字典
        figures_dir: 可视化图表目录
        pycaret_figures_dir: PyCaret图表目录
    """

    models: Dict[str, Any]
    weights: Dict[str, float]
    feature_columns: List[str]
    defaults: Dict[str, Any]
    figures_dir: str
    pycaret_figures_dir: str


class ModelRepository:
    """
    模型仓库：负责加载配置、读取指标、筛选模型并完成加载。
    """

    def __init__(self) -> None:
        # 加载API配置
        config_loader = create_config_loader()
        self.api_config = config_loader.load_config("api_config")

        # 读取配置路径
        paths = self.api_config["paths"]
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.metrics_csv = self._resolve_path(paths["metrics_csv"])
        self.models_dir = self._resolve_path(paths["models_dir"])
        self.training_data = self._resolve_path(paths["training_data"])
        self.figures_dir = self._resolve_path(paths["figures_dir"])
        self.pycaret_figures_dir = self._resolve_path(paths["pycaret_figures_dir"])

        # 读取训练数据与模型指标
        self.training_df = self._load_training_data()
        self.metrics_df = self._load_metrics()

    def build_bundle(self) -> ModelBundle:
        """
        构建完整的模型包，供预测器使用。
        """
        feature_columns = [c for c in self.training_df.columns if c != "CVD"]
        defaults = self._compute_defaults(self.training_df, feature_columns)
        models, weights = self._load_models()

        if not models:
            raise ApiError("未加载到任何可用模型，请检查模型文件与筛选条件。", 500)

        return ModelBundle(
            models=models,
            weights=weights,
            feature_columns=feature_columns,
            defaults=defaults,
            figures_dir=self.figures_dir,
            pycaret_figures_dir=self.pycaret_figures_dir,
        )

    def _load_training_data(self) -> pd.DataFrame:
        """
        读取训练数据（用于获取特征列表与默认值）。
        """
        if not os.path.exists(self.training_data):
            raise ApiError(f"训练数据文件不存在: {self.training_data}", 500)
        return pd.read_csv(self.training_data)

    def _load_metrics(self) -> pd.DataFrame:
        """
        读取模型评估指标（AUC、Recall等），用于筛选模型。
        """
        if not os.path.exists(self.metrics_csv):
            raise ApiError(f"模型指标文件不存在: {self.metrics_csv}", 500)
        return pd.read_csv(self.metrics_csv)

    def _resolve_path(self, path_value: str) -> str:
        """
        将配置中的路径转换为绝对路径。

        规则：
        - 如果已经是绝对路径，直接返回
        - 如果是相对路径，则以项目根目录为基准拼接
        """
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(self.project_root, path_value))

    def _infer_is_categorical(self, series: pd.Series) -> bool:
        """
        根据数据分布推断该列是否为分类型特征。
        规则：
        - object 类型直接视为分类
        - 数值型但唯一值较少（<=10）也视为分类
        """
        if series.dtype == object:
            return True
        unique_count = series.dropna().nunique()
        return unique_count <= 10

    def _compute_defaults(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        根据训练数据计算各字段的默认值。
        """
        defaults_cfg = self.api_config["defaults"]
        defaults: Dict[str, Any] = {}

        for col in feature_columns:
            series = df[col]
            is_categorical = self._infer_is_categorical(series)

            if is_categorical:
                defaults[col] = self._coerce_value(self._get_mode(series))
            else:
                numeric_strategy = defaults_cfg.get("numeric", "median")
                if numeric_strategy == "mean":
                    defaults[col] = float(series.mean())
                elif numeric_strategy == "mode":
                    defaults[col] = self._coerce_value(self._get_mode(series))
                else:
                    defaults[col] = float(series.median())

        return defaults

    def _get_mode(self, series: pd.Series) -> Any:
        """
        获取众数，若为空则返回0作为兜底值。
        """
        mode = series.mode()
        if not mode.empty:
            return mode.iloc[0]
        return 0

    def _coerce_value(self, value: Any) -> Any:
        """
        将numpy标量等类型转换为Python原生类型，方便JSON序列化。
        """
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def _select_models(self) -> pd.DataFrame:
        """
        按配置筛选模型，返回过滤后的指标DataFrame。
        """
        cfg = self.api_config["model_selection"]
        df = self.metrics_df.copy()

        # 统一列名，避免大小写差异导致问题
        df.columns = [c.strip() for c in df.columns]

        # 白名单优先
        allow = cfg.get("allow_models", [])
        if allow:
            df = df[df["Model"].isin(allow)]
        else:
            df = df[(df["AUC"] >= cfg.get("auc_threshold", 0.0)) & (df["Recall"] >= cfg.get("recall_min", 0.0))]

        # 黑名单过滤
        deny = cfg.get("deny_models", [])
        if deny:
            df = df[~df["Model"].isin(deny)]

        return df

    def _build_model_filename(self, model_name: str) -> str:
        """
        根据指标文件中的模型名称生成模型文件名。
        规则：pycaret_{model_name小写}_model.pkl
        """
        safe_name = model_name.lower()
        return f"pycaret_{safe_name}_model.pkl"

    def _load_models(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        加载模型并计算权重。
        """
        df = self._select_models()
        if df.empty:
            raise ApiError("模型筛选后为空，请调整阈值或检查指标文件。", 500)

        models: Dict[str, Any] = {}
        weights: Dict[str, float] = {}

        weight_by = self.api_config["ensemble"].get("weight_by", "auc")
        for _, row in df.iterrows():
            model_name = row["Model"]
            filename = self._build_model_filename(model_name)
            model_path = os.path.join(self.models_dir, filename)

            if not os.path.exists(model_path):
                # 若模型文件缺失，直接跳过并继续处理其他模型
                continue

            models[model_name] = joblib.load(model_path)

            # 权重策略：目前仅支持按AUC加权
            if weight_by == "auc":
                weights[model_name] = float(row["AUC"])
            else:
                weights[model_name] = 1.0

        # 权重归一化
        if self.api_config["ensemble"].get("normalize", True) and weights:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return models, weights
