# AGENTS 变更记录

本文件用于记录本仓库的关键变更，便于回溯与协作沟通。

## 变更记录
- 2026-02-03：初始化 `src/api/__init__.py`，准备API模块目录结构。

- 2026-02-03：新增 `configs/api_config.yaml`，用于API模型筛选与输入配置。

- 2026-02-03：新增 `src/api/schemas.py`，定义预测请求与响应结构。

- 2026-02-03：新增 `src/api/errors.py`，集中处理API异常与统一错误响应。

- 2026-02-03：新增 `src/api/model_loader.py`，实现模型与配置的统一加载。

- 2026-02-03：新增 `src/api/predictor.py`，实现模型推理与集成逻辑。
