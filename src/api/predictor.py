# -*- coding: utf-8 -*-
# 本文件负责推理逻辑：
# 1) 输入特征补全与校验
# 2) 调用多个模型进行预测
# 3) 按权重进行集成
# 4) 可选返回可视化资源列表

from __future__ import annotations

from typing import Dict, Any, List, Optional
import os

import numpy as np
import pandas as pd

from src.api.errors import ApiError
from src.api.model_loader import ModelBundle


class PredictService:
    """
    预测服务类，将模型加载结果封装为可复用的预测能力。

    Args:
        bundle: 已加载好的模型与配置集合
        allow_unknown_fields: 是否允许输入中出现未知字段
    """

    def __init__(self, bundle: ModelBundle, allow_unknown_fields: bool) -> None:
        self.bundle = bundle
        self.allow_unknown_fields = allow_unknown_fields

    def predict(self, input_features: Dict[str, Any], return_viz: bool) -> Dict[str, Any]:
        """
        对外预测入口。

        Args:
            input_features: 用户输入的特征字典（可只包含部分字段）
            return_viz: 是否返回可视化资源列表

        Returns:
            预测结果字典（包含最终标签、概率、子模型详情、权重等）
        """
        # 校验字段并构造模型输入
        feature_df = self._build_feature_row(input_features)

        # 逐个模型预测概率
        model_details: Dict[str, float] = {}
        failed_models: Dict[str, str] = {}

        for name, model in self.bundle.models.items():
            try:
                model_details[name] = self._predict_proba(model, feature_df)
            except ApiError as exc:
                # 记录失败原因，但不中断整体预测
                failed_models[name] = exc.details.get("error", exc.message)

        if not model_details:
            raise ApiError("无可用模型完成预测。", 500)

        # 仅对成功模型进行权重归一化
        used_weights = self._normalize_weights(
            {name: self.bundle.weights.get(name, 1.0) for name in model_details.keys()}
        )

        # 按权重求加权平均
        probability = self._ensemble_probability(model_details, used_weights)
        label = int(probability >= 0.5)

        figures = None
        if return_viz:
            figures = self._collect_figures()

        result = {
            "label": label,
            "probability": probability,
            "model_details": model_details,
            "used_models": used_weights,
            "figures": figures,
        }
        if failed_models:
            result["failed_models"] = failed_models
        return result

    def _build_feature_row(self, input_features: Dict[str, Any]) -> pd.DataFrame:
        """
        将输入特征补全为模型所需的完整特征行。

        处理流程：
        1) 检查未知字段
        2) 以默认值为基底
        3) 用输入特征覆盖默认值
        4) 保证列顺序与训练一致
        """
        feature_columns = self.bundle.feature_columns

        if not self.allow_unknown_fields:
            unknown = [k for k in input_features.keys() if k not in feature_columns]
            if unknown:
                raise ApiError(
                    message="输入包含未知字段",
                    status_code=400,
                    details={"unknown_fields": unknown},
                )

        # 用默认值初始化
        row = dict(self.bundle.defaults)

        # 覆盖用户提供的值（只覆盖非None字段）
        for key, value in input_features.items():
            if value is not None:
                row[key] = value

        # 构造DataFrame并保证列顺序
        try:
            feature_df = pd.DataFrame([row], columns=feature_columns)
        except Exception as exc:
            raise ApiError("构造模型输入失败", 500, {"error": str(exc)})

        return feature_df

    def _predict_proba(self, model: Any, feature_df: pd.DataFrame) -> float:
        """
        获取单个模型的正类概率。

        兼容策略：
        - 优先尝试使用 PyCaret 的 predict_model（用于加载的PyCaret Pipeline）
        - 优先使用 predict_proba
        - 其次使用 decision_function + sigmoid
        - 最后使用 predict 结果作为概率兜底
        """
        try:
            # 1) PyCaret推理路径（若可用）
            try:
                from pycaret.classification import predict_model as pycaret_predict_model
                pred_df = pycaret_predict_model(model, data=feature_df)
                # 常见的概率列名：Score_1 或 Score
                for col in ["Score_1", "Score", "prediction_score"]:
                    if col in pred_df.columns:
                        return float(pred_df.loc[0, col])
            except Exception:
                # 如果PyCaret不可用或预测失败，继续走通用路径
                pass

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_df)
                return float(proba[0][1])

            if hasattr(model, "decision_function"):
                score = model.decision_function(feature_df)
                # sigmoid 将任意实数映射到 (0,1)
                return float(1 / (1 + np.exp(-score)))[0]

            pred = model.predict(feature_df)
            return float(pred[0])
        except Exception as exc:
            raise ApiError("模型预测失败", 500, {"error": str(exc)})

    def _ensemble_probability(self, model_details: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        对多个模型的概率做加权平均。
        """
        total = 0.0
        weight_sum = 0.0
        for name, prob in model_details.items():
            weight = weights.get(name, 1.0)
            total += prob * weight
            weight_sum += weight

        if weight_sum == 0:
            # 若权重全部为0，则退化为简单平均
            return float(sum(model_details.values()) / len(model_details))

        return float(total / weight_sum)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        对权重进行归一化，确保总和为1。
        """
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _collect_figures(self) -> List[str]:
        """
        收集可视化资源路径，返回可直接用于前端的URL列表。
        """
        figures: List[str] = []

        # 读取静态根目录（用于生成相对路径）
        static_root = self.bundle.static_root
        if not static_root:
            return figures
        if not os.path.exists(static_root):
            return figures

        # 将本地路径转换为URL路径，保持与StaticFiles挂载一致
        def to_url(file_path: str) -> str:
            # 统一使用static_root作为静态根目录
            rel = os.path.relpath(file_path, start=static_root)
            # 将Windows路径分隔符转换为URL风格
            rel = rel.replace("\\", "/")
            return f"/static/{rel}"

        for root_dir in [self.bundle.figures_dir, self.bundle.pycaret_figures_dir]:
            if not os.path.exists(root_dir):
                continue
            for base, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(".png"):
                        figures.append(to_url(os.path.join(base, file)))

        return figures
