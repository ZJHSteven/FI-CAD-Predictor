# 项目状态快照（保持短小）

## 当前结论（必须最新）
- 现状：CHARLS 五个波次已经按 RAW、extracted、curated、audit 四层整理完毕，第一版纵向建模代码已开始落地，目标是 2011 基线预测 2013/2015/2018/2020 新发心脏病相关事件。
- 已完成：补入 2011 家户问卷文档、2013 构建支出收入财富数据、2015 血检候选包；删除 2011 重复包和 `.part` 残留；重建 81 张 `.dta`、81 份 CSV、81 份 Parquet、81 份 metadata JSON；curated 层中文目录、metadata 内部路径和官网文档副本已同步；新增 `src.fi_cad` 建模入口和配置骨架。
- 正在做：运行第一版数据集构建、训练、评估、报告生成和回归测试，确认是否存在全判负、召回率过低、FPR/FNR 异常或数据泄露风险。
- 下一步：根据真实训练指标决定是否调整 FI 缺陷项、阈值策略、类别权重或特征集；不要为了“看起来分数高”而修改终点口径。

## 关键决策与理由（防止“吃书”）
- 决策A：目录统一使用 `2011-wave1` 这种命名，保留年份和波次信息。原因：既符合用户习惯，也方便后续按年份或波次检索。
- 决策B：CSV 不是唯一输出，必须同时保留 `.dta` 原件、Parquet 和元数据 JSON。原因：CSV 对 Stata 的标签、缺失码和部分类型信息不够完整，Parquet 适合后续建模读取，metadata JSON 保存标签语义。
- 决策C：`对照清单.md` 是唯一验收真源，旧 `data/MANIFEST.md` 不再使用。原因：旧 manifest 曾经基于错误环境和过期判断生成，不能继续作为数据完整性依据。
- 决策D：curated 层使用中文表目录，但 CSV 列名仍保留 Stata 原始变量名。原因：原始列名是跨问卷、codebook、DTA 和后续脚本对齐的稳定键；变量解释放在 metadata 和后续变量字典中。
- 决策E：第一版论文终点写作“心脏病相关事件”，不硬写成纯 CAD。原因：CHARLS 慢病条目本身覆盖 heart attack / coronary heart disease / angina / congestive heart failure / other heart problems。
- 决策F：第一版主模型默认不纳入血检增强特征。原因：主链路先验证问卷+体测模型，且 2015 血检 RAW SHA1 仍需复核；血检增强模型后续再单独标注风险。

## 常见坑 / 复现方法
- 坑1：不要再用旧 manifest 判断 2013 总包是否损坏；当前 `data/raw/2013-wave2/CHARLS2013_Dataset.zip` SHA1 已命中清单，且能解出 18 张总包表。
- 坑2：2015 年 `Blood.zip` 能解出 `Blood.dta`，但外层 SHA1 不等于清单中的血检数据 SHA1，不能表述成 RAW SHA1 完全闭合。
- 坑3：Stata 特殊缺失码会造成同一列同时有数字和字符串；Parquet 输出会把这类 object 列转成字符串，并在 metadata 的 `parquet_string_columns` 记录。
- 坑4：`data/extracted` 只允许 `.dta`，如果补充压缩包里夹带文档，脚本会清掉 extracted 里的非 DTA 文件；文档真源仍在 RAW，curated 只保存便于查看的副本。
- 坑5：如果只重命名 curated 目录、不更新 metadata 里的 `csv_file/parquet_file/metadata_file`，后续脚本会读到不存在的旧路径；测试已覆盖这个问题。
- 坑6：训练必须先按 ID 切分，再在 Pipeline 内 fit 缺失填补、标准化和模型；不能先全量填补/标准化/筛选后再切分。
- 坑7：AUC 高不等于模型可用；如果阈值下几乎全判负、召回率接近 0 或 FNR 很高，必须把 run 标成诊断问题。
