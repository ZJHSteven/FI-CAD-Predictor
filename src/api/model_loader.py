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
        feature_columns: 模型特征列名列表（来自配置）
        defaults: 各特征的默认值字典（来自配置）
        figures_dir: 可视化图表目录（可选）
        pycaret_figures_dir: PyCaret图表目录（可选）
        static_root: 静态资源根目录（用于FastAPI静态挂载）
    """

    models: Dict[str, Any]
    weights: Dict[str, float]
    feature_columns: List[str]
    defaults: Dict[str, Any]
    figures_dir: str
    pycaret_figures_dir: str
    static_root: str


class FixedFeatureSelector:
    """
    固定特征选择器（兜底方案）。

    当PyCaret的feature_selection步骤损坏时，用此选择器保证维度一致。
    该选择器只保留输入的前N个特征，避免模型因特征数不匹配报错。
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def fit(self, X: Any, y: Any = None) -> "FixedFeatureSelector":
        # 无需训练，直接返回自身
        return self

    def transform(self, X: Any) -> Any:
        # 同时兼容DataFrame与numpy数组
        if hasattr(X, "iloc"):
            return X.iloc[:, : self.n_features]
        return X[:, : self.n_features]


class NamedFeatureSelector:
    """
    按特征名选择列的选择器（更精确的兜底方案）。

    优先使用训练好的模型 feature_names_in_ 来匹配正确列名，
    确保推理阶段的特征与训练阶段一致。
    """

    def __init__(self, feature_names: list, input_names: list | None = None) -> None:
        self.feature_names = list(feature_names)
        self.input_names = list(input_names) if input_names is not None else None

    def fit(self, X: Any, y: Any = None) -> "NamedFeatureSelector":
        # 无需训练，直接返回自身
        return self

    def transform(self, X: Any) -> Any:
        # 如果是DataFrame，直接按列名选择
        if hasattr(X, "loc"):
            return X.loc[:, self.feature_names]

        # 如果是numpy数组，则尝试用input_names构造DataFrame再选择
        if self.input_names is not None:
            import pandas as pd
            X_df = pd.DataFrame(X, columns=self.input_names)
            return X_df.loc[:, self.feature_names].values

        # 再次兜底：按特征数量裁剪
        return X[:, : len(self.feature_names)]


class ModelRepository:
    """
    模型仓库：负责加载配置、读取指标、筛选模型并完成加载。
    """

    def __init__(self) -> None:
        # ===== 配置加载 =====
        # 使用统一配置加载器读取API配置，确保部署环境仅依赖配置即可运行
        config_loader = create_config_loader()  # 创建配置加载器
        self.api_config = config_loader.load_config("api_config")  # 读取api_config.yaml

        # ===== 路径配置 =====
        # 读取可选路径配置，部署版允许路径为空
        paths = self.api_config.get("paths", {})  # 获取路径配置（可为空字典）
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # 项目根目录
        self.models_dir = self._resolve_optional_path(paths.get("models_dir"))  # 模型目录（可为空）
        self.figures_dir = self._resolve_optional_path(paths.get("figures_dir"))  # 图表目录（可为空）
        self.pycaret_figures_dir = self._resolve_optional_path(paths.get("pycaret_figures_dir"))  # PyCaret图表目录（可为空）

        # ===== 特征与默认值 =====
        # 部署版不再读取训练数据，而是直接从配置获取特征列表与默认值
        self.feature_columns, self.defaults = self._load_feature_schema()

    def build_bundle(self) -> ModelBundle:
        """
        构建完整的模型包，供预测器使用。
        """
        # 加载模型并获取权重
        models, weights = self._load_models()  # 读取模型与权重配置

        if not models:
            raise ApiError("未加载到任何可用模型，请检查模型文件与筛选条件。", 500)

        return ModelBundle(
            models=models,
            weights=weights,
            feature_columns=self.feature_columns,
            defaults=self.defaults,
            figures_dir=self.figures_dir,
            pycaret_figures_dir=self.pycaret_figures_dir,
            static_root=self._get_static_root(),
        )

    def _get_static_root(self) -> str:
        """
        获取静态资源根目录。

        说明：
        - 如果配置了 figures_dir，则将其作为静态根目录
        - 未配置时返回空字符串，表示不挂载静态资源
        """
        if self.figures_dir:
            return self.figures_dir
        return ""

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

    def _resolve_optional_path(self, path_value: str | None) -> str:
        """
        处理可选路径配置。

        当配置缺失或为空字符串时，返回空字符串以表示“未启用”。
        """
        if not path_value:
            return ""
        return self._resolve_path(path_value)

    def _load_feature_schema(self) -> Tuple[List[str], Dict[str, Any]]:
        """
        从配置中读取特征列与默认值。

        Returns:
            (feature_columns, defaults)
        """
        features_cfg = self.api_config.get("features")  # 读取features配置块
        if not features_cfg:
            raise ApiError("缺少features配置，请在api_config.yaml中补充。", 500)

        feature_columns = features_cfg.get("columns")  # 获取特征列列表
        defaults_cfg = features_cfg.get("defaults")  # 获取默认值字典

        if not feature_columns:
            raise ApiError("features.columns不能为空。", 500)
        if defaults_cfg is None:
            raise ApiError("features.defaults不能为空。", 500)

        # 检查默认值是否覆盖所有特征列
        missing = [col for col in feature_columns if col not in defaults_cfg]
        if missing:
            raise ApiError(
                "默认值配置缺失，请补全features.defaults。",
                500,
                {"missing_fields": missing},
            )

        # 将默认值整理为按特征列顺序的字典
        defaults: Dict[str, Any] = {}
        for col in feature_columns:
            defaults[col] = self._coerce_value(defaults_cfg[col])

        return feature_columns, defaults

    def _coerce_value(self, value: Any) -> Any:
        """
        将numpy标量等类型转换为Python原生类型，方便JSON序列化。
        """
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def _load_models(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        加载模型并计算权重。
        """
        models_cfg = self.api_config.get("models", [])  # 读取模型配置列表
        if not models_cfg:
            raise ApiError("未配置models列表，请在api_config.yaml中补充。", 500)

        models: Dict[str, Any] = {}  # 已加载模型字典
        weights: Dict[str, float] = {}  # 模型权重字典
        missing_files: List[str] = []  # 记录缺失文件，便于排查

        # 尝试优先使用PyCaret的load_model（可确保Pipeline与预处理一致）
        try:
            from pycaret.classification import load_model as pycaret_load_model
        except Exception:
            pycaret_load_model = None

        for item in models_cfg:
            model_name = item.get("name")  # 模型名称
            model_path_cfg = item.get("path")  # 模型路径
            model_weight = float(item.get("weight", 1.0))  # 权重，默认1.0

            if not model_name or not model_path_cfg:
                raise ApiError(
                    "模型配置缺少name或path字段。",
                    500,
                    {"bad_item": item},
                )

            # 将路径转换为绝对路径
            model_path = self._resolve_path(model_path_cfg)

            if not os.path.exists(model_path):
                missing_files.append(model_path)
                continue

            if pycaret_load_model is not None:
                # PyCaret的load_model通常传入不带扩展名的路径
                base_path = os.path.splitext(model_path)[0]
                models[model_name] = pycaret_load_model(base_path)
            else:
                # 兜底：使用joblib直接加载
                models[model_name] = joblib.load(model_path)

            # 尝试修复feature_selection步骤（若缺少support_会导致预测失败）
            self._repair_feature_selection(models[model_name])

            # 保存权重配置
            weights[model_name] = model_weight

        if not models:
            raise ApiError(
                "模型文件缺失或未能加载，请检查models配置。",
                500,
                {"missing_files": missing_files},
            )

        # 权重归一化（保持与旧版一致的行为）
        if self.api_config.get("ensemble", {}).get("normalize", True) and weights:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return models, weights

    def _repair_feature_selection(self, model: Any) -> None:
        """
        修复PyCaret Pipeline中feature_selection步骤的拟合状态。

        背景：
        部分已保存的PyCaret模型在加载后，feature_selection内部的
        SelectFromModel可能未包含support_，会导致transform时报错。

        修复策略：
        1) 检测Pipeline是否包含feature_selection步骤
        2) 用训练数据通过“feature_selection之前的预处理步骤”生成X
        3) 对SelectFromModel执行fit，生成support_
        """
        try:
            if not hasattr(model, "steps"):
                return

            step_dict = dict(model.steps)
            if "feature_selection" not in step_dict:
                return

            fs_wrapper = step_dict["feature_selection"]
            transformer = getattr(fs_wrapper, "transformer", None)
            if transformer is None:
                return

            # 如果已经存在support_，说明已拟合，无需处理
            if hasattr(transformer, "support_"):
                return

            # ===== 无训练数据时的修复策略 =====
            # 部署版不再保留训练数据，因此直接走“已训练模型兜底”逻辑
            trained_model = step_dict.get("trained_model")  # 尝试获取已训练模型
            if trained_model is None:
                return

            if not (hasattr(trained_model, "coef_") or hasattr(trained_model, "feature_importances_")):
                return

            # 强制使用已训练模型作为重要性来源
            transformer.estimator_ = trained_model
            transformer.prefit = True

            # 使用特征名选择器或固定特征数选择器兜底
            feature_names = getattr(trained_model, "feature_names_in_", None)
            input_names = self.feature_columns  # 使用配置中的特征顺序
            n_features = getattr(trained_model, "n_features_in_", None)

            if feature_names is not None:
                selector = NamedFeatureSelector(feature_names, input_names)
            elif n_features is not None:
                selector = FixedFeatureSelector(n_features)
            else:
                return

            model.steps = [
                (name, step) if name != "feature_selection" else ("feature_selection", selector)
                for name, step in model.steps
            ]
        except Exception:
            # 若修复失败，保持原状，交由预测阶段处理失败
            return
