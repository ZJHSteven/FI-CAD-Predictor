# FI-CAD-Predictor 执行计划

## 当前未完成
1. 跑通第一版纵向建模全链路：`python -m src.fi_cad.build_dataset` -> `python -m src.fi_cad.train --config configs/modeling.yaml` -> `python -m src.fi_cad.evaluate --run latest` -> `python -m src.fi_cad.make_report --run latest`。
2. 复核真实训练结果：如果某模型出现几乎全判负、召回率过低、FPR/FNR 异常或校准明显偏离，必须先写诊断，不要把“训练完成”当成“模型可用”。
3. 复核 2015 年“血检数据”RAW SHA1：当前 `data/raw/2015-wave3/Blood.zip` 可解出 `Blood.dta`，但 SHA1 是 `e350b371ffd6f72967f502e269c6a85e37b22407`，没有命中对照清单中的 `de5b1d80e9b72d61b690dff19b191f50f21ee8c5`。

## 验收标准
- `对照清单.md` 中列出的 2011、2013、2015、2018、2020 文档和数据条目，在 RAW 层都能找到对应文件或总包覆盖证据。
- 除 2015 血检候选包外，所有带 SHA1 的 RAW 条目都能命中清单或由总包解出的 DTA 覆盖；不能有 `.part`、重复下载副本或放错年份目录的文件。
- 所有需要建模的数据都能解压成 `.dta`，并能被读取至少元数据和首行数据。
- curated 层每个 `.dta` 至少对应一个 CSV 和一个 metadata JSON；如果环境支持 Parquet，还要输出 Parquet。
- curated 层每个波次都有 `文档` 目录；明确针对单表的文档也应复制到对应表目录下的 `文档` 子目录。
- 测试覆盖清单解析、RAW 完整性、解压完整性、元数据提取和异常失败路径。
- 中文化目录重命名完成后，`data/curated/<wave>` 下面只保留中文模块目录名，metadata 内部路径必须指向真实存在的 CSV、Parquet 和 JSON。
- 建模测试必须覆盖：基线心脏病阳性排除、ID 切分不重叠、指标表包含 FPR/FNR/Recall/Precision/F1/AUC。
- 训练 run 必须写出 `run_manifest.json`、模型指标表、阈值表、模型文件和核心图表。
