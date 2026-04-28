# CHARLS 数据清单

## 目录约定
- `data/raw/<year-waveX>`: 原始下载文件，保留压缩包和原始文档。
- `data/extracted/<year-waveX>`: 把原始压缩包解出来后的 `.dta` 原件。
- `data/curated/<year-waveX>`: 从 `.dta` 导出的 CSV 和元数据 JSON。
- 目录命名统一使用 `2011-wave1` 这种形式，保留年份和波次信息。

## 当前状态总览
| 波次 | 原始目录 | 文件数 | 文档 | 压缩包正常 | 压缩包警告 | 压缩包错误 | 已解压 DTA | CSV | 元数据 | 错误报告 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011 年 wave1 | `D:/Workspace/FI-CAD-Predictor/data/raw/2011-wave1` | 28 | 4 | 23 | 0 | 0 | 36 | 36 | 36 | 0 |
| 2013 年 wave2 | `D:/Workspace/FI-CAD-Predictor/data/raw/2013-wave2` | 6 | 5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2015 年 wave3 | `D:/Workspace/FI-CAD-Predictor/data/raw/2015-wave3` | 6 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| 2018 年 wave4 | `D:/Workspace/FI-CAD-Predictor/data/raw/2018-wave4` | 6 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| 2020 年 wave5 | `D:/Workspace/FI-CAD-Predictor/data/raw/2020-wave5` | 5 | 4 | 1 | 0 | 0 | 10 | 10 | 10 | 0 |

## 盘点说明
- 下面的表格记录的是当前工作区里已经落地的原始文件，而不是官网所有理论上存在的条目。
- 如果某个压缩包能列出内部文件但 7-Zip 报了头部错误，会标成 `warn`；如果完全打不开，会标成 `error`。
- CSV 导出会保留数值和字符串内容，但 Stata 的标签和一些细节类型仍建议结合 `.dta` 原件和元数据 JSON。

## 2011 年 wave1

- 原始目录：`D:/Workspace/FI-CAD-Predictor/data/raw/2011-wave1`
- 解压目录：`D:/Workspace/FI-CAD-Predictor/data/extracted/2011-wave1`
- 可读化目录：`D:/Workspace/FI-CAD-Predictor/data/curated/2011-wave1`
- 统计：28 个文件，4 个文档，23 个正常压缩包，0 个带警告压缩包，0 个错误压缩包，1 个疑似未完成下载。
- 解压与导出：36 个 `.dta`，36 个 CSV，36 个元数据 JSON，0 个错误报告。

| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |
| --- | --- | ---: | --- | --- | --- |
| `biomarkers.rar` | archive | 953515 | `03ab2d3a1fb67813f00aa57e26147efa6590f395` | ok | biomarkers.dta |
| `Blood_20140429.zip` | archive | 504675 | `86587c9eeb7bbcb84e2ddbf95d17365f59704a12` | ok | Blood_20140429.dta |
| `blood_user_guide_en_20140429.pdf` | document | 116927 | `51cb0cb21cf4f14b92130932788aba8b00cc2a5a` | ok | 说明文档或问卷 |
| `CHARLS_codebook.rar` | archive | 1692696 | `f9f4b2a7d72675182b33035eedbac6f6dc8dc0b3` | ok | CHARLS_codebook.pdf |
| `child.zip` | archive | 425258 | `9efccd64251500b468357c57e1f364e3f0b2f0a5` | ok | child.dta |
| `Chinese_users__guide_20130407_.pdf` | document | 1405577 | `41dde3c19bed8f3864e4f06068949223601e099f` | ok | 说明文档或问卷 |
| `community.rar` | archive | 199098 | `4d4acf6c10f775e8af0acca7ecdd417214700fb4` | ok | community.dta |
| `Community_questionnaire_C_and_E__20130312_.pdf` | document | 386604 | `54ba9005eef06ca6e38a0aac1dfa033e35de51c1` | ok | 说明文档或问卷 |
| `demographic_background.rar` | archive | 443745 | `9a4afebf04e9060e5211a1fa6c2bc850f5d08a72` | ok | demographic_background.dta |
| `exp_income_wealth.zip` | archive | 600889 | `e73a9c1827978a597dad8469fa1a2866b4944da6` | ok | exp_income_wealth.dta |
| `family_information.rar` | archive | 1120995 | `bd2daa4f918cd73fd20e1415cb71f84274a8ad48` | ok | family_information.dta |
| `family_transfer.rar` | archive | 313738 | `9227d4b7f584ecc4cd65dd202e8b70541c0b8698` | ok | family_transfer.dta |
| `health_care_and_insurance.rar` | archive | 454750 | `f512be857ea0578f710dfab6ba3fd4a20f2a2d73` | ok | health_care_and_insurance.dta |
| `health_status_and_functioning(1).rar` | archive | 1147001 | `03d08b0a90a4da65187d28c0a943cb5693bbab52` | ok | health_status_and_functioning.dta |
| `health_status_and_functioning.rar` | archive | 1147001 | `03d08b0a90a4da65187d28c0a943cb5693bbab52` | ok | health_status_and_functioning.dta |
| `hhmember.zip` | archive | 266914 | `b281c4e94f8390f49d17ab7565f69a2dcb64ea2c` | ok | hhmember.dta |
| `household_and_community_questionnaire_data.rar` | archive | 7100502 | `5bb36f9f37a7ea7ca9e93ba4d083d1306c280105` | ok | household_income.dta；household_roster.dta；housing_characteristics.dta；individual_income.dta；interviewer_observation.dta；psu.dta；weight.dta；work_retirement_and_pension.dta；biomarkers.dta；community.dta；demographic_background.dta；family_information.dta；family_transfer.dta；health_care_and_insurance.dta；health_status_and_functioning.dta |
| `household_income.rar` | archive | 832912 | `e568e05be18d4aba6daf8bad00f99b9d39f61a0f` | ok | household_income.dta |
| `household_roster.rar` | archive | 312047 | `cf9df8a22370c801f46ec2ea6e3f93f61b542584` | ok | household_roster.dta |
| `housing_characteristics.rar` | archive | 126608 | `b613e4d51b03421312c9f5cbb1649a32e56492e4` | ok | housing_characteristics.dta |
| `individual_income.rar` | archive | 237316 | `164083aeee16e828be573926aa9d6a11772d9235` | ok | individual_income.dta |
| `interviewer_observation.rar` | archive | 111850 | `abda5266fd2cf0882d859d46c44f85ce01903fae` | ok | interviewer_observation.dta |
| `Medical-questionnaire-2011.doc` | document | 765440 | `652087926efcacb2abf796371b1a2b9dbbfefee5` | ok | 说明文档或问卷 |
| `parent.zip` | archive | 329999 | `30d232bf98e4e5b6eaa4448af20be0a3c41ba538` | ok | parent.dta |
| `PSU.zip` | archive | 8810 | `74c78392b668375937e78ce5da4f06f5ba83f0f9` | ok | PSU.dta |
| `TxaXB7q8.zip.part` | partial | 12315286 | `7517d597d2eb5abfa91d58f3cd03aa9845defc85` | warn | 未完成下载或分卷残留 |
| `weight.rar` | archive | 242091 | `e514672f071f4aec6e7012031967ca53f1b13d19` | ok | weight.dta |
| `work_retirement_and_pension.rar` | archive | 598601 | `e00bce7ae8a5e36a497cf96d9cf72cbe7823513c` | ok | work_retirement_and_pension.dta |

## 2013 年 wave2

- 原始目录：`D:/Workspace/FI-CAD-Predictor/data/raw/2013-wave2`
- 解压目录：`D:/Workspace/FI-CAD-Predictor/data/extracted/2013-wave2`
- 可读化目录：`D:/Workspace/FI-CAD-Predictor/data/curated/2013-wave2`
- 统计：6 个文件，5 个文档，0 个正常压缩包，0 个带警告压缩包，1 个错误压缩包，0 个疑似未完成下载。
- 解压与导出：0 个 `.dta`，0 个 CSV，0 个元数据 JSON，0 个错误报告。

| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |
| --- | --- | ---: | --- | --- | --- |
| `CHARLS2013_Dataset.zip` | archive | 0 | `da39a3ee5e6b4b0d3255bfef95601890afd80709` | error | 1 个错误；无法作为正常压缩包读取；ERROR: D:\Workspace\FI-CAD-Predictor\data\raw\2013-wave2\CHARLS2013_Dataset.zip : Cannot open the file as archive |
| `CHARLS_Wave2_Biomarker_Questionnaire.pdf` | document | 511801 | `7599805448e07e75680e6c15d92f4d7d78d886b2` | ok | 说明文档或问卷 |
| `CHARLS_Wave2_CodeBook.pdf` | document | 1794165 | `c9c5714dbffee02050343f9b100aba501050b7a3` | ok | 说明文档或问卷 |
| `CHARLS_Wave2_Exit_VA_Questionnaire.pdf` | document | 812194 | `6ec50048ef1d0f23f6b63b9fb0fc937f484cd621` | ok | 说明文档或问卷 |
| `CHARLS_Wave2_Main_Questionnaire.pdf` | document | 1719074 | `d1132a289bb3a37b955ebc3f9d42d56dc072a1e4` | ok | 说明文档或问卷 |
| `CHARLS_Wave2_Release_Note.pdf` | document | 36679 | `8f4a90c3dbadb49711333ab1e689bf2756583932` | ok | 说明文档或问卷 |

## 2015 年 wave3

- 原始目录：`D:/Workspace/FI-CAD-Predictor/data/raw/2015-wave3`
- 解压目录：`D:/Workspace/FI-CAD-Predictor/data/extracted/2015-wave3`
- 可读化目录：`D:/Workspace/FI-CAD-Predictor/data/curated/2015-wave3`
- 统计：6 个文件，5 个文档，0 个正常压缩包，1 个带警告压缩包，0 个错误压缩包，0 个疑似未完成下载。
- 解压与导出：1 个 `.dta`，0 个 CSV，0 个元数据 JSON，1 个错误报告。

| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |
| --- | --- | ---: | --- | --- | --- |
| `CHARLS2015r.zip` | archive | 13399645 | `d84c5ffd9e9b4b1b9224ba1f39e660920d81f35a` | warn | Biomarker.dta |
| `CHARLS_2015_Biomarker_Questionnaire.pdf` | document | 518656 | `df292d012cf75cce536727e9ec62e67b454f9946` | ok | 说明文档或问卷 |
| `CHARLS_2015_Blood_Data_Release_Note.pdf` | document | 167102 | `f48c549a7f1253230d4c11cbae1687966a3d39d5` | ok | 说明文档或问卷 |
| `CHARLS_2015_Codebook.pdf` | document | 1641489 | `92e413c887c946a9d5c939cab736ff96d05fc020` | ok | 说明文档或问卷 |
| `CHARLS_2015_Questionnaire.pdf` | document | 1607798 | `dffe2db15d6e13cbe5499d5bf2556b17cdc8c729` | ok | 说明文档或问卷 |
| `CHARLS_2015_Release_Note.pdf` | document | 33596 | `05f2d352f856680830cbac7b044488f07b0f7538` | ok | 说明文档或问卷 |

## 2018 年 wave4

- 原始目录：`D:/Workspace/FI-CAD-Predictor/data/raw/2018-wave4`
- 解压目录：`D:/Workspace/FI-CAD-Predictor/data/extracted/2018-wave4`
- 可读化目录：`D:/Workspace/FI-CAD-Predictor/data/curated/2018-wave4`
- 统计：6 个文件，5 个文档，0 个正常压缩包，1 个带警告压缩包，0 个错误压缩包，0 个疑似未完成下载。
- 解压与导出：1 个 `.dta`，0 个 CSV，0 个元数据 JSON，1 个错误报告。

| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |
| --- | --- | ---: | --- | --- | --- |
| `CHARLS-HCAP-User-Guidance-cn.pdf` | document | 216352 | `47079d5f1b4da50dd038f0244ec33dd918fc0f78` | ok | 说明文档或问卷 |
| `CHARLS2018r.zip` | archive | 11942202 | `f7b00282a1dfe0459798103a681338d387ac48b3` | warn | Cognition.dta |
| `CHARLS_2018_Codebook.pdf` | document | 1505502 | `8f8017cec22518e0251cf15ee3a700c9142a7a14` | ok | 说明文档或问卷 |
| `CHARLS_2018_Household_Questionnaire.pdf` | document | 2045907 | `6c547dde4240eae5f7fffb22556d26109186b974` | ok | 说明文档或问卷 |
| `CHARLS_2018_Users_Guide.pdf` | document | 327091 | `18d92656677464d10096377e1864db59bd4daa51` | ok | 说明文档或问卷 |
| `w2018-data-alert-cn.pdf` | document | 155366 | `f0be35e7d65b4bbb6938cf908fb7bb8e64238018` | ok | 说明文档或问卷 |

## 2020 年 wave5

- 原始目录：`D:/Workspace/FI-CAD-Predictor/data/raw/2020-wave5`
- 解压目录：`D:/Workspace/FI-CAD-Predictor/data/extracted/2020-wave5`
- 可读化目录：`D:/Workspace/FI-CAD-Predictor/data/curated/2020-wave5`
- 统计：5 个文件，4 个文档，1 个正常压缩包，0 个带警告压缩包，0 个错误压缩包，0 个疑似未完成下载。
- 解压与导出：10 个 `.dta`，10 个 CSV，10 个元数据 JSON，0 个错误报告。

| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |
| --- | --- | ---: | --- | --- | --- |
| `CHARLS2020r.zip` | archive | 8534006 | `fd73eb699cbacbfaaff1c55bcdba11b23eaa0736` | ok | COVID_Module.dta；Demographic_Background.dta；Exit_Module.dta；Family_Information.dta；Health_Status_and_Functioning.dta；Household_Income.dta；Individual_Income.dta；Sample_Infor.dta；Weights.dta；Work_Retirement.dta |
| `CHARLS_2020_Codebook.pdf` | document | 918582 | `bf669aa916daf7da68e32357d1e08f31a3a0dae7` | ok | 说明文档或问卷 |
| `CHARLS_2020_Flowchart_Chinese.pdf` | document | 1462940 | `83da535e9542ad6974ff3474b7fe0fcded0ed0bd` | ok | 说明文档或问卷 |
| `CHARLS_2020_Questionnaire_Chinese.pdf` | document | 1065938 | `307fb0f902e76fd08e07222324a4ce5fa24fec99` | ok | 说明文档或问卷 |
| `CHARLS_2020_User_Guide_Chinese.pdf` | document | 416238 | `9bf1e673136f375beb097e88b6018bf88855f17b` | ok | 说明文档或问卷 |

## 后续落地目录
- `data/extracted`：放解压后的原始 Stata 文件。
- `data/curated`：放 CSV 和元数据 JSON，后面建模优先读这里。
- 如果后续再补下载，只要重新跑整理脚本即可自动刷新清单。
