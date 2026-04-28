"""CHARLS 数据整理流水线测试。

这些测试不再沿用旧的“外层文件数”判断方式，而是围绕当前真实验收口径：
1. `对照清单.md` 是唯一清单真源。
2. RAW 层要能按 SHA1 或总包覆盖证据解释每个条目。
3. extracted 层只保留 `.dta` 表格文件。
4. curated 层每张表都有 CSV、Parquet、metadata JSON。
5. metadata JSON 必须能保存变量标签和值标签，不能只剩裸表。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from charls_data import (  # noqa: E402
    AUDIT_ROOT,
    CHECKLIST_PATH,
    CURATED_ROOT,
    EXTRACTED_ROOT,
    RAW_ROOT,
    build_raw_file_index,
    compute_sha1,
    list_archive_members,
    parse_checklist,
)


class CharlsDataPipelineTests(unittest.TestCase):
    """验证当前数据资产整理结果。"""

    def test_parse_checklist_keeps_official_entries(self) -> None:
        """清单解析必须覆盖官网整理后的所有列表项。"""

        entries = parse_checklist(REPO_ROOT / CHECKLIST_PATH)
        self.assertEqual(len(entries), 111)
        self.assertTrue(any(item.year == "2013" and item.item == "死因信息" for item in entries))
        self.assertTrue(any(item.year == "2015" and item.item == "血检数据" for item in entries))

    def test_known_raw_sha1_values_match_checklist(self) -> None:
        """几个关键 RAW 包的 SHA1 必须和对照清单一致。"""

        expected = {
            "data/raw/2013-wave2/CHARLS2013_Dataset.zip": "7517d597d2eb5abfa91d58f3cd03aa9845defc85",
            "data/raw/2013-wave2/exp_income_wealth.zip": "8488faf9c5273afbce1b073ce5929513cd538512",
            "data/raw/2015-wave3/CHARLS2015r.zip": "47aab04178a6895ce687beb2f74286215e8f6b33",
            "data/raw/2018-wave4/CHARLS2018r.zip": "42f3e6bf26cea8072d9fa9f6050f413556d20a69",
            "data/raw/2020-wave5/CHARLS2020r.zip": "fd73eb699cbacbfaaff1c55bcdba11b23eaa0736",
        }
        for relative, sha1_hex in expected.items():
            self.assertEqual(compute_sha1(REPO_ROOT / relative), sha1_hex)

    def test_archive_members_cover_expected_tables(self) -> None:
        """总包和补充包内部必须能列出关键 DTA。"""

        wave2_members = set(list_archive_members(REPO_ROOT / "data/raw/2013-wave2/CHARLS2013_Dataset.zip"))
        wave3_blood_members = set(list_archive_members(REPO_ROOT / "data/raw/2015-wave3/Blood.zip"))

        self.assertIn("Verbal_Autopsy.dta", wave2_members)
        self.assertIn("PSU.dta", wave2_members)
        self.assertIn("Blood.dta", wave3_blood_members)

    def test_extracted_layer_contains_only_dta_files(self) -> None:
        """extracted 是表格层，不能混入 PDF/DOC 等文档。"""

        expected_counts = {
            "2011-wave1": 20,
            "2013-wave2": 19,
            "2015-wave3": 18,
            "2018-wave4": 14,
            "2020-wave5": 10,
        }
        for wave, expected_count in expected_counts.items():
            files = [path for path in (REPO_ROOT / EXTRACTED_ROOT / wave).rglob("*") if path.is_file()]
            dta_files = [path for path in files if path.suffix.lower() == ".dta"]
            self.assertEqual(len(dta_files), expected_count)
            self.assertEqual(len(files), expected_count)

    def test_curated_layer_has_csv_parquet_and_metadata(self) -> None:
        """每张 DTA 表都必须有 CSV、Parquet 和 metadata JSON。"""

        for dta_path in (REPO_ROOT / EXTRACTED_ROOT).rglob("*.dta"):
            relative = dta_path.relative_to(REPO_ROOT / EXTRACTED_ROOT)
            table_dir = REPO_ROOT / CURATED_ROOT / relative.parent / relative.stem
            self.assertTrue((table_dir / f"{relative.stem}.csv").exists(), msg=str(dta_path))
            self.assertTrue((table_dir / f"{relative.stem}.parquet").exists(), msg=str(dta_path))
            self.assertTrue((table_dir / f"{relative.stem}.metadata.json").exists(), msg=str(dta_path))

    def test_metadata_preserves_labels(self) -> None:
        """metadata 里必须保留变量标签和值标签。"""

        metadata_path = REPO_ROOT / "data/curated/2013-wave2/Health_Status_and_Functioning/Health_Status_and_Functioning.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertGreater(metadata["row_count"], 0)
        self.assertGreater(metadata["column_count"], 0)
        self.assertGreater(len(metadata["column_names_to_labels"]), 0)
        self.assertGreater(len(metadata["value_labels"]), 0)

    def test_audit_has_no_missing_sha1_but_records_blood_review(self) -> None:
        """当前唯一未闭合项应是 2015 血检 RAW SHA1 复核。"""

        audit = json.loads((REPO_ROOT / AUDIT_ROOT / "charls_data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["coverage_summary"]["missing_sha1"], 0)
        self.assertEqual(audit["coverage_summary"]["candidate_dta_but_sha_mismatch"], 1)

        needs_review = [
            item
            for item in audit["coverage"]
            if item["status"] == "candidate_dta_but_sha_mismatch"
        ]
        self.assertEqual(len(needs_review), 1)
        self.assertEqual(needs_review[0]["year"], "2015")
        self.assertEqual(needs_review[0]["item"], "血检数据")

    def test_raw_index_does_not_contain_known_dirty_files(self) -> None:
        """RAW 层不能再出现明确的重复包和 `.part` 残留。"""

        raw_files = build_raw_file_index(REPO_ROOT / RAW_ROOT)
        raw_paths = {Path(item["path"]).name for item in raw_files}

        self.assertNotIn("health_status_and_functioning(1).rar", raw_paths)
        self.assertNotIn("TxaXB7q8.zip.part", raw_paths)


if __name__ == "__main__":
    unittest.main()
