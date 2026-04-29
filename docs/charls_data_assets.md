# CHARLS 数据资产说明

## 当前验收口径

本项目以仓库根目录的 `对照清单.md` 作为唯一验收真源。  
`data/audit/charls_data_audit.json` 是机器可读的验收结果，后续人员如果要确认数据是否齐全，应优先查看这个 JSON，而不是旧的 `data/MANIFEST.md`。

当前结论：
- 2011、2013、2015、2018、2020 五个波次的数据都已经进入 `data/raw`、`data/extracted`、`data/curated`。
- `data/extracted` 当前共有 81 张 `.dta` 表，且不混入 PDF/DOC 文档。
- `data/curated` 中每张 `.dta` 都对应 3 个输出：CSV、Parquet、metadata JSON；每个波次还同步了官网文档副本。
- 唯一需要人工复核的 RAW 项：2015 年“血检数据”。本地 `data/raw/2015-wave3/Blood.zip` 可以解出 `Blood.dta`，但外层 SHA1 是 `e350b371ffd6f72967f502e269c6a85e37b22407`，没有命中 `对照清单.md` 中的 `de5b1d80e9b72d61b690dff19b191f50f21ee8c5`。

## 目录结构

`data/raw/<year-waveX>` 保存官网下载的原始文件。  
这一层用于审计和复现，不做内容改写。压缩包、PDF、DOC 都保留在这里。

`data/extracted/<year-waveX>` 保存从 RAW 压缩包解出的 Stata `.dta` 文件。  
这一层只保存表格文件，不保存文档。2011 年官网同时提供总包和分包，为避免重复表，当前 extracted 使用分包解压结果；2013 年之后优先使用总包，再补充总包没有覆盖的构建/血检数据。

`data/curated/<year-waveX>/<table>` 保存从 `.dta` 提取出的可用版本：
- `<table>.csv`：给人工检查和通用工具读取。
- `<table>.parquet`：给后续建模高效读取。
- `<table>.metadata.json`：保存 Stata 标签、值标签、原始类型、缺失值定义等语义信息。

`data/curated/<year-waveX>/文档` 保存该波次官网文档副本。  
如果文档明显只服务某一类表，脚本会额外复制到 `data/curated/<year-waveX>/<table>/文档`，例如血检发布说明会进入“血检数据/文档”，体检问卷会进入“体检信息/文档”。RAW 仍然是原始下载真源，curated 文档只是便于分析时就近查看。

`data/audit/charls_data_audit.json` 保存验收摘要：
- `raw_files`：RAW 文件路径、大小、SHA1、压缩包内部成员。
- `extraction_results`：每个被解压压缩包的结果。
- `curated_results`：每张表的 CSV、Parquet、metadata 输出路径。
- `document_results`：每个 RAW 文档同步到了哪些 curated 文档目录。
- `coverage`：`对照清单.md` 每个条目的覆盖状态。
- `curated_summary_by_wave`：每个波次的表数、行数、列数、标签数和值标签数。

## 标签和值标签在哪里

Stata 文件不只是裸表，还包含变量标签和值标签。当前提取策略如下：
- CSV 和 Parquet 保留原始编码值，不把编码直接替换成中文或英文标签。
- 变量标签保存在 metadata JSON 的 `column_names_to_labels` 和 `column_labels`。
- 值标签保存在 metadata JSON 的 `value_labels`、`variable_to_label` 和 `variable_value_labels`。
- Stata 原始类型保存在 `original_variable_types` 和 `readstat_variable_types`。
- Stata 用户自定义缺失值信息保存在 `missing_user_values` 和 `missing_ranges`。

注意：Parquet 要求单列类型稳定。对于同时包含数字和 Stata 特殊缺失码的列，脚本会把该列在 Parquet 中转成字符串，并把列名记录在 metadata JSON 的 `parquet_string_columns` 中。CSV 和原始 `.dta` 仍保留可复核信息。

## 如何重新生成

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\update_charls_data.py
```

成功时输出应包含：
- `curated 表：81`
- `导出错误：0`
- `同步文档：25 个 RAW 文档条目`
- `需复核条目：1`

当前这 1 个复核条目是 2015 年血检数据 SHA1 不匹配。除非重新下载到 SHA1 完全匹配的官方包，否则不要把它描述成“RAW SHA1 完全闭合”。

## 后续建模读取建议

训练模型时优先读 Parquet：

```python
import pandas as pd

frame = pd.read_parquet("data/curated/2013-wave2/健康状况与功能/Health_Status_and_Functioning.parquet")
```

如果需要解释变量含义或把编码值翻译成标签，再读取同目录 metadata：

```python
import json
from pathlib import Path

metadata = json.loads(Path("data/curated/2013-wave2/健康状况与功能/Health_Status_and_Functioning.metadata.json").read_text(encoding="utf-8"))
variable_labels = metadata["column_names_to_labels"]
value_labels = metadata["value_labels"]
```

如果要做最严格的数据追溯，回到 `data/extracted` 的 `.dta` 原件和 `data/raw` 的原始压缩包。
