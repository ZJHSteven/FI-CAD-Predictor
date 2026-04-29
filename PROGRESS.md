# 项目状态快照（保持短小）

## 当前结论（必须最新）
- 现状：CHARLS 五个波次已经按 RAW、extracted、curated、audit 四层整理完毕，第一版纵向建模代码已开始落地，目标是 2011 基线预测 2013/2015/2018/2020 新发心脏病相关事件。
- 已完成：补入 2011 家户问卷文档、2013 构建支出收入财富数据、2015 血检候选包；删除 2011 重复包和 `.part` 残留；重建 81 张 `.dta`、81 份 CSV、81 份 Parquet、81 份 metadata JSON；curated 层中文目录、metadata 内部路径和官网文档副本已同步；新增 `src.fi_cad` 建模入口和配置骨架。
- 正在做：进入第二轮建模改进；已确认上一版 `height_cm_2011` 错把中位数约 43 的 `qh006` 当作身高优先来源，且当前 `fi_2011` 已改回旧论文/旧分支可复核的 11 项等权 FI。
- 下一步：必须重新训练模型 run；旧 run `output/runs/20260429-204720` 的 FI、身高和特征重要性解释已经过期，只能作为错误诊断参考。

## 关键决策与理由（防止“吃书”）
- 决策A：目录统一使用 `2011-wave1` 这种命名，保留年份和波次信息。原因：既符合用户习惯，也方便后续按年份或波次检索。
- 决策B：CSV 不是唯一输出，必须同时保留 `.dta` 原件、Parquet 和元数据 JSON。原因：CSV 对 Stata 的标签、缺失码和部分类型信息不够完整，Parquet 适合后续建模读取，metadata JSON 保存标签语义。
- 决策C：`对照清单.md` 是唯一验收真源，旧 `data/MANIFEST.md` 不再使用。原因：旧 manifest 曾经基于错误环境和过期判断生成，不能继续作为数据完整性依据。
- 决策D：curated 层使用中文表目录，但 CSV 列名仍保留 Stata 原始变量名。原因：原始列名是跨问卷、codebook、DTA 和后续脚本对齐的稳定键；变量解释放在 metadata 和后续变量字典中。
- 决策E：第一版论文终点写作“心脏病相关事件”，不硬写成纯 CAD。原因：CHARLS 慢病条目本身覆盖 heart attack / coronary heart disease / angina / congestive heart failure / other heart problems。
- 决策F：第一版主模型默认不纳入血检增强特征。原因：主链路先验证问卷+体测模型，且 2015 血检 RAW SHA1 仍需复核；血检增强模型后续再单独标注风险。
- 决策G：`fi_2011` 使用旧论文/旧分支 11 项等权口径：高血压、慢性肺病、哮喘、胃/消化疾病、关节炎/风湿、髋部骨折、精神疾病、糖尿病、癌症、肾病、BMI_to_FI。原因：这是旧分支 `2011年FI+CVD及变量新.csv` 可还原出的明确公式，心脏病和卒中不进入 FI，避免结局泄露。

## 常见坑 / 复现方法
- 坑1：不要再用旧 manifest 判断 2013 总包是否损坏；当前 `data/raw/2013-wave2/CHARLS2013_Dataset.zip` SHA1 已命中清单，且能解出 18 张总包表。
- 坑2：2015 年 `Blood.zip` 能解出 `Blood.dta`，但外层 SHA1 不等于清单中的血检数据 SHA1，不能表述成 RAW SHA1 完全闭合。
- 坑3：Stata 特殊缺失码会造成同一列同时有数字和字符串；Parquet 输出会把这类 object 列转成字符串，并在 metadata 的 `parquet_string_columns` 记录。
- 坑4：`data/extracted` 只允许 `.dta`，如果补充压缩包里夹带文档，脚本会清掉 extracted 里的非 DTA 文件；文档真源仍在 RAW，curated 只保存便于查看的副本。
- 坑5：如果只重命名 curated 目录、不更新 metadata 里的 `csv_file/parquet_file/metadata_file`，后续脚本会读到不存在的旧路径；测试已覆盖这个问题。
- 坑6：训练必须先按 ID 切分，再在 Pipeline 内 fit 缺失填补、标准化和模型；不能先全量填补/标准化/筛选后再切分。
- 坑7：AUC 高不等于模型可用；如果阈值下几乎全判负、召回率接近 0 或 FNR 很高，必须把 run 标成诊断问题。
- 坑8：2011 Wave1 的 ID 常见为 11 位，2013 之后常见为 12 位；建模前必须统一为 12 位，否则随访观察会被误判为 0。
- 坑9：当前第一版模型已经避免全判负，但 Precision 约 0.19~0.21、FPR 可到 0.42，说明误报多；后续不能只汇报“训练成功”，必须继续做特征工程和终点复核。
- 坑10：`qh006` 在当前 2011 体测表里中位数约 43，不是成人身高；身高应使用更符合厘米身高分布的 `qi002`，否则特征重要性会被错误变量名误导。
- 坑11：旧分支指标不要误记成 AUC 0.9；`output/results/pycaret_model_metrics.csv` 显示 Logistic/GBDT/LDA 等 AUC 约 0.77，但该旧流程仍有横断面和泄露风险，只能用来恢复 FI 定义，不能当作当前纵向模型效果。

## 最近验证
- `python -m unittest discover -s tests -p "test*.py" -v`：14 项通过。
- `python -m compileall src tests`：通过。
- `python -m src.fi_cad.build_dataset --config configs/modeling.yaml`：生成 14785 人建模表，阳性率 0.1340，特征数 52。
- `python -m src.fi_cad.train --config configs/modeling.yaml`：最新 run `output/runs/20260429-204720`，Logistic、RandomForest、XGBoost、LightGBM、CatBoost 均完成训练。
- `python -m src.fi_cad.evaluate --config configs/modeling.yaml --run latest`：指标表已生成，五模型均 warning。
- `python -m src.fi_cad.make_report --config configs/modeling.yaml --run latest`：报告已生成。
- `.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -v`：16 项通过。
- `.venv\Scripts\python.exe -m src.fi_cad.build_dataset --config configs/modeling.yaml`：基于旧论文 11 项 FI 重新生成 14785 人建模表，阳性率 0.1340，特征数 57；`height_cm_2011` 中位数约 157.9，`fi_2011` 均值约 0.126。
