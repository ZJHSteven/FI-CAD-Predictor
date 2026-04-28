# 项目状态快照（保持短小）

## 当前结论（必须最新）
- 现状：CHARLS 五个波次已经按 RAW、extracted、curated、audit 四层整理完毕，`data/MANIFEST.md` 已删除，后续以 `data/audit/charls_data_audit.json` 为机器可读验收结果。
- 已完成：补入 2011 家户问卷文档、2013 构建支出收入财富数据、2015 血检候选包；删除 2011 重复包和 `.part` 残留；重建 81 张 `.dta`、81 份 CSV、81 份 Parquet、81 份 metadata JSON。
- 正在做：只剩 2015 年“血检数据”RAW SHA1 复核；本地 `Blood.zip` 可解出 `Blood.dta`，但 SHA1 没有命中对照清单。
- 下一步：如果能重新下载到 SHA1 为 `de5b1d80e9b72d61b690dff19b191f50f21ee8c5` 的官方 2015 血检包，就替换 `data/raw/2015-wave3/Blood.zip` 并重跑 `scripts/update_charls_data.py`。

## 关键决策与理由（防止“吃书”）
- 决策A：目录统一使用 `2011-wave1` 这种命名，保留年份和波次信息。原因：既符合用户习惯，也方便后续按年份或波次检索。
- 决策B：CSV 不是唯一输出，必须同时保留 `.dta` 原件、Parquet 和元数据 JSON。原因：CSV 对 Stata 的标签、缺失码和部分类型信息不够完整，Parquet 适合后续建模读取，metadata JSON 保存标签语义。
- 决策C：`对照清单.md` 是唯一验收真源，旧 `data/MANIFEST.md` 不再使用。原因：旧 manifest 曾经基于错误环境和过期判断生成，不能继续作为数据完整性依据。

## 常见坑 / 复现方法
- 坑1：不要再用旧 manifest 判断 2013 总包是否损坏；当前 `data/raw/2013-wave2/CHARLS2013_Dataset.zip` SHA1 已命中清单，且能解出 18 张总包表。
- 坑2：2015 年 `Blood.zip` 能解出 `Blood.dta`，但外层 SHA1 不等于清单中的血检数据 SHA1，不能表述成 RAW SHA1 完全闭合。
- 坑3：Stata 特殊缺失码会造成同一列同时有数字和字符串；Parquet 输出会把这类 object 列转成字符串，并在 metadata 的 `parquet_string_columns` 记录。
- 坑4：`data/extracted` 只允许 `.dta`，如果补充压缩包里夹带文档，脚本会清掉 extracted 里的非 DTA 文件，文档只保留在 RAW。
