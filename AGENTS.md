# AGENTS 变更记录

本文件用于记录本仓库的关键变更，便于回溯与协作沟通。

## 变更记录
- 2026-02-03：初始化 `src/api/__init__.py`，准备API模块目录结构。

- 2026-02-03：新增 `configs/api_config.yaml`，用于API模型筛选与输入配置。

- 2026-02-03：新增 `src/api/schemas.py`，定义预测请求与响应结构。

- 2026-02-03：新增 `src/api/errors.py`，集中处理API异常与统一错误响应。

- 2026-02-03：新增 `src/api/model_loader.py`，实现模型与配置的统一加载。

- 2026-02-03：新增 `src/api/predictor.py`，实现模型推理与集成逻辑。

- 2026-02-03：新增 `src/api/app.py`，实现FastAPI应用入口与路由。

- 2026-02-03：更新 `src/api/schemas.py`，允许保留未知字段以便后端校验。

- 2026-02-03：更新 `requirements.txt`，新增FastAPI与Uvicorn依赖。

- 2026-02-03：新增 `run_api.bat`，用于启动FastAPI服务。

- 2026-02-03：修正 `run_api.bat` 路径转义问题。

- 2026-02-03：更新 `README.md`，补充API服务使用说明。

- 2026-02-03：更新 `src/api/model_loader.py`，统一将路径解析为绝对路径。

- 2026-02-03：更新 `src/api/model_loader.py`，补充默认值类型转换以便JSON序列化。

- 2026-02-03：更新 `src/api/predictor.py`，修正静态资源URL路径计算。

- 2026-02-04：删除 `requirements.txt` 并清理旧虚拟环境目录（venv/.venv）。
