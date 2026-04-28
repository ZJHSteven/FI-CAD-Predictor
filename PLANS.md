# FI-CAD-Predictor 后续计划

## 当前未完成
1. 重新下载或补齐 `2013-wave2/CHARLS2013_Dataset.zip`。当前 `data/raw/2013-wave2` 里还是 0 字节占位，仓库根目录另有一个不匹配网站 SHA1 的额外 zip。
2. 继续调查 `2015-wave3/Biomarker.dta` 和 `2018-wave4/Cognition.dta` 的读取问题。现在脚本会保留 `.dta` 原件并写错误报告，但还没找到更强的转换后端。

## 已完成，不再重复写进计划
- 已完成原始目录统一命名为 `2011-wave1` 到 `2020-wave5`。
- 已完成 `data/raw`、`data/extracted`、`data/curated` 三层目录和占位文件。
- 已完成 `data/MANIFEST.md` 自动生成。
- 已完成可执行脚本和单元测试。
- 已完成把 `对照清单.md` 整理成按年份分层的 Markdown 清单。
