# -*- coding: utf-8 -*-
# 本文件是FastAPI应用的入口，负责：
# 1) 应用初始化
# 2) 加载模型与预测服务
# 3) 注册路由与异常处理
# 4) 挂载静态资源目录（用于图表访问）

from __future__ import annotations

import os
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.schemas import PredictRequest, PredictResponse
from src.api.errors import ApiError, api_error_handler
from src.api.model_loader import ModelRepository
from src.api.predictor import PredictService


def _get_project_root() -> str:
    """
    获取项目根目录的绝对路径。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ===== 应用初始化 =====
app = FastAPI(title="FI/CVD 预测 API", version="0.1.0")
app.add_exception_handler(ApiError, api_error_handler)

# ===== 静态资源挂载 =====
# 说明：将 output 目录挂载为 /static，便于前端访问图表
project_root = _get_project_root()
output_dir = os.path.join(project_root, "output")
if os.path.exists(output_dir):
    app.mount("/static", StaticFiles(directory=output_dir), name="static")

# ===== 预测服务初始化 =====
repository = ModelRepository()
bundle = repository.build_bundle()
allow_unknown_fields = repository.api_config["input"].get("allow_unknown_fields", False)
predict_service = PredictService(bundle=bundle, allow_unknown_fields=allow_unknown_fields)


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    健康检查接口。
    返回已加载模型数量及权重信息，便于确认服务状态。
    """
    return {
        "status": "ok",
        "model_count": len(bundle.models),
        "weights": bundle.weights,
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    """
    返回当前参与集成的模型列表与权重。
    """
    return {
        "models": list(bundle.models.keys()),
        "weights": bundle.weights,
    }


@app.get("/features")
def features() -> Dict[str, Any]:
    """
    返回模型特征列表与默认值，便于前端自动生成表单。
    """
    return {
        "required_fields": repository.api_config["input"].get("required_fields", []),
        "feature_columns": bundle.feature_columns,
        "defaults": bundle.defaults,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> Dict[str, Any]:
    """
    预测接口。

    输入：PredictRequest
    输出：PredictResponse
    """
    input_features = request.to_feature_dict()
    result = predict_service.predict(input_features, return_viz=request.return_viz)
    return result
