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

- 2026-02-04：使用 `uv init` 初始化 `pyproject.toml`。

- 2026-02-04：设置 `.python-version` 为 3.10 以兼容 PyCaret。

- 2026-02-04：更新 `pyproject.toml`，将 Python 版本要求调整为 3.10。

- 2026-02-04：使用 `uv add` 添加 numpy 依赖，开始迁移依赖管理。

- 2026-02-04：使用 `uv add` 补充其余依赖（移除pickle5以避免编译失败）。

- 2026-02-04：更新 `setup_env.bat`，改为使用UV管理环境与依赖。

- 2026-02-04：更新 `run_project.bat`，改为使用UV运行主流程。

- 2026-02-04：更新 `run_api.bat`，改为使用UV启动API服务。

- 2026-02-04：更新 `README.md`，改为UV环境与运行说明。

- 2026-02-04：使用 `uv add` 新增 httpx 以支持 TestClient 测试。

- 2026-02-04：修复 `src/api/predictor.py` 的路径拼接，避免反斜杠导致语法错误。

- 2026-02-04：使用 `uv remove` 移除 pycaret（避免与numpy版本冲突）。

- 2026-02-04：使用 `uv add` 升级 numpy 至 1.24.3 以匹配已训练模型。

- 2026-02-04：重新通过 `uv add` 安装 pycaret（版本 3.0.0）以支持模型加载。

- 2026-02-04：使用 `uv add` 升级 scikit-learn 至 1.3.2 以兼容已训练模型。

- 2026-02-04：更新 `src/api/predictor.py` 与 `src/api/schemas.py`，允许跳过失败模型并返回失败原因。

- 2026-02-04：更新 `src/api/model_loader.py` 与 `src/api/predictor.py`，改用PyCaret加载与预测。

- 2026-02-04：更新 `src/api/model_loader.py`，尝试修复feature_selection的拟合状态。

- 2026-02-04：更新 `src/api/model_loader.py`，补齐LightGBM的silent属性以完成特征选择拟合。

- 2026-02-04：完善 `src/api/model_loader.py` 的特征选择修复，失败时回退到已训练模型。

- 2026-02-04：更新 `src/api/model_loader.py`，加入固定特征选择器兜底逻辑。

- 2026-02-04：更新 `src/api/model_loader.py`，使用特征名选择器确保列匹配。

- 2026-02-04：更新 `src/api/app.py`，加入CORS配置以便前端访问。

- 2026-02-04：使用Vite创建 `frontend` React TS 最小项目结构。
