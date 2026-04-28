"""CHARLS 数据整理工具测试。

这些测试尽量覆盖三类风险：
1. 目录命名迁移是否正确。
2. 真实压缩包是否能被识别、列出、解压。
3. `.dta` 导出 CSV 和元数据是否能跑通。

说明：
- 测试默认优先使用仓库里的真实样本文件，这样能更早发现环境问题。
- 如果 7-Zip 不可用，相关测试会跳过，而不是报出假失败。
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS_ROOT))

from charls_data import (  # noqa: E402
    WAVE_SPECS,
    convert_stata_tree,
    extract_archive,
    find_7z_executable,
    inspect_archive,
    normalize_raw_layout,
    run_pipeline,
)


class CharlsDataTests(unittest.TestCase):
    """围绕数据整理脚本的集成测试。"""

    def test_wave_specs_are_in_chronological_order(self) -> None:
        """波次定义必须按时间顺序排列，不能乱。"""

        years = [spec.year for spec in WAVE_SPECS]
        waves = [spec.wave for spec in WAVE_SPECS]
        self.assertEqual(years, sorted(years))
        self.assertEqual(waves, [1, 2, 3, 4, 5])
        self.assertEqual(WAVE_SPECS[0].slug, "2011-wave1")
        self.assertEqual(WAVE_SPECS[-1].slug, "2020-wave5")

    def test_normalize_raw_layout_renames_legacy_directories(self) -> None:
        """旧的年份目录应被改成规范化目录名。"""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            legacy = data_root / "raw" / "2011"
            legacy.mkdir(parents=True)
            (legacy / "sample.txt").write_text("ok", encoding="utf-8")

            moved = normalize_raw_layout(data_root)

            self.assertEqual(len(moved), 1)
            self.assertFalse(legacy.exists())
            self.assertTrue((data_root / "raw" / "2011-wave1").exists())
            self.assertTrue((data_root / "raw" / "2011-wave1" / "sample.txt").exists())

    def test_scan_real_wave1_archive(self) -> None:
        """真实的 wave1 压缩包应该能被正常识别。"""

        data_root = REPO_ROOT / "data"
        wave1 = _find_wave_dir(data_root, "2011-wave1", "2011")
        archive = wave1 / "biomarkers.rar"
        seven_zip = find_7z_executable()

        inspection = inspect_archive(archive, seven_zip)

        self.assertEqual(inspection.status, "ok")
        self.assertEqual(inspection.inner_files, ["biomarkers.dta"])

    def test_scan_real_empty_archive_reports_error(self) -> None:
        """零字节压缩包必须被明确标成错误。"""

        data_root = REPO_ROOT / "data"
        wave2 = _find_wave_dir(data_root, "2013-wave2", "2013")
        archive = wave2 / "CHARLS2013_Dataset.zip"
        seven_zip = find_7z_executable()

        inspection = inspect_archive(archive, seven_zip)

        self.assertEqual(inspection.status, "error")
        self.assertEqual(inspection.inner_files, [])

    def test_scan_real_warning_archive_reports_warn(self) -> None:
        """带头部异常但仍能列出文件的压缩包，应当标成 warn。"""

        data_root = REPO_ROOT / "data"
        wave3 = _find_wave_dir(data_root, "2015-wave3", "2015")
        archive = wave3 / "CHARLS2015r.zip"
        seven_zip = find_7z_executable()

        inspection = inspect_archive(archive, seven_zip)

        self.assertEqual(inspection.status, "warn")
        self.assertIn("Biomarker.dta", inspection.inner_files)

    def test_extract_and_convert_real_biomarkers_dta(self) -> None:
        """一个真实的 `.rar` 样本应当能完整走到 CSV 导出。"""

        data_root = REPO_ROOT / "data"
        wave1 = _find_wave_dir(data_root, "2011-wave1", "2011")
        archive = wave1 / "biomarkers.rar"
        seven_zip = find_7z_executable()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            extract_root = tmp_root / "extracted"
            curated_root = tmp_root / "curated"
            ok, message = extract_archive(archive, extract_root, seven_zip)
            self.assertTrue(ok, msg=message)

            dta_files = list(extract_root.rglob("*.dta"))
            self.assertEqual(len(dta_files), 1)

            results = convert_stata_tree(extract_root, curated_root)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "ok")
            self.assertIsNotNone(results[0].csv_path)
            self.assertIsNotNone(results[0].meta_path)
            self.assertTrue(results[0].csv_path and results[0].csv_path.exists())
            self.assertTrue(results[0].meta_path and results[0].meta_path.exists())

            meta = json.loads(results[0].meta_path.read_text(encoding="utf-8"))
            self.assertTrue(meta["source_file"].endswith("biomarkers.dta"))
            self.assertGreater(meta["row_count"], 0)
            self.assertGreater(meta["column_count"], 0)

    def test_extract_warning_archive_still_outputs_file(self) -> None:
        """带警告的压缩包只要能吐出文件，就不应该直接判死。"""

        data_root = REPO_ROOT / "data"
        wave3 = _find_wave_dir(data_root, "2015-wave3", "2015")
        archive = wave3 / "CHARLS2015r.zip"
        seven_zip = find_7z_executable()

        with tempfile.TemporaryDirectory() as tmp:
            extract_root = Path(tmp) / "extracted"
            ok, message = extract_archive(archive, extract_root, seven_zip)
            self.assertTrue(ok, msg=message)
            self.assertGreater(len(list(extract_root.rglob("*.dta"))), 0)

    def test_pipeline_writes_manifest(self) -> None:
        """完整管线至少要能把清单写出来。"""

        data_root = REPO_ROOT / "data"
        result = run_pipeline(data_root, skip_extract=True, skip_convert=True)
        manifest_path = result["manifest_path"]
        self.assertTrue(Path(manifest_path).exists())
        content = Path(manifest_path).read_text(encoding="utf-8")
        self.assertIn("# CHARLS 数据清单", content)
        self.assertIn("2011 年 wave1", content)
        self.assertIn("2013 年 wave2", content)


def _find_wave_dir(data_root: Path, canonical: str, legacy: str) -> Path:
    """兼容已经迁移和未迁移两种布局。"""

    candidates = [data_root / "raw" / canonical, data_root / "raw" / legacy]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"在 {data_root} 下找不到 {canonical} / {legacy}")


if __name__ == "__main__":
    unittest.main()
