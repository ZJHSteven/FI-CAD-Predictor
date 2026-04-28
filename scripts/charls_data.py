"""CHARLS 数据资产整理与验收工具。

本文件负责把“官网下载清单”和“本地数据目录”连成一条可复核的证据链：
1. RAW 层：检查 `data/raw` 里的原始文件 SHA1、压缩包可读性、脏文件。
2. extracted 层：按人工确认的压缩包计划解压出 `.dta`，避免重复包造成重复表。
3. curated 层：把 `.dta` 转成 CSV、Parquet 和 metadata JSON。
4. audit 层：写出机器可读的验收摘要，供测试、文档和后续建模复用。

重要设计取舍：
- `对照清单.md` 是唯一的本地验收真源；脚本只解析它，不再相信旧的 MANIFEST。
- CSV 只解决“人能看表”的问题，不保存 Stata 标签；标签和缺失值信息必须写入 JSON。
- Parquet 用于后续建模高效读取，但 `.dta` 原件仍然保留，避免任何转换损失不可逆。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyreadstat


CHUNK_SIZE = 1024 * 1024
CHECKLIST_PATH = Path("对照清单.md")
DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
EXTRACTED_ROOT = DATA_ROOT / "extracted"
CURATED_ROOT = DATA_ROOT / "curated"
AUDIT_ROOT = DATA_ROOT / "audit"


@dataclass(frozen=True)
class WaveSpec:
    """描述一个 CHARLS 主调查波次。

    字段说明：
    - year: 官网清单里的年份。
    - wave: 项目内部使用的波次编号。
    - slug: 本仓库统一目录名。
    """

    year: str
    wave: int
    slug: str


@dataclass(frozen=True)
class ChecklistEntry:
    """从 `对照清单.md` 中解析出的一个条目。"""

    year: str
    section: str
    item: str
    sha1_hex: str | None


@dataclass(frozen=True)
class ExpectedTable:
    """官网数据条目与实际 `.dta` 文件之间的对应关系。

    字段说明：
    - year / item: 对应 `对照清单.md` 的数据下载条目。
    - dta_name: 解压后应该出现的 Stata 数据文件名。
    - source: `bundle` 表示由“以下所有数据的压缩包”覆盖；`direct` 表示应该有单独 RAW 包。
    """

    year: str
    item: str
    dta_name: str
    source: str


WAVES: tuple[WaveSpec, ...] = (
    WaveSpec("2011", 1, "2011-wave1"),
    WaveSpec("2013", 2, "2013-wave2"),
    WaveSpec("2015", 3, "2015-wave3"),
    WaveSpec("2018", 4, "2018-wave4"),
    WaveSpec("2020", 5, "2020-wave5"),
)


# 这里写的是“本仓库决定解压哪些包”，不是官网全部 RAW 文件清单。
# 2011 同时有总包和分包；为了 extracted 层不重复，只解压分包。
# 2013 之后主要用总包，再额外解压总包没有覆盖的构建/血检数据。
EXTRACTION_PLAN: dict[str, tuple[str, ...]] = {
    "2011-wave1": (
        "biomarkers.rar",
        "Blood_20140429.zip",
        "child.zip",
        "community.rar",
        "demographic_background.rar",
        "exp_income_wealth.zip",
        "family_information.rar",
        "family_transfer.rar",
        "health_care_and_insurance.rar",
        "health_status_and_functioning.rar",
        "hhmember.zip",
        "household_income.rar",
        "household_roster.rar",
        "housing_characteristics.rar",
        "individual_income.rar",
        "interviewer_observation.rar",
        "parent.zip",
        "PSU.zip",
        "weight.rar",
        "work_retirement_and_pension.rar",
    ),
    "2013-wave2": (
        "CHARLS2013_Dataset.zip",
        "exp_income_wealth.zip",
    ),
    "2015-wave3": (
        "CHARLS2015r.zip",
        "Blood.zip",
    ),
    "2018-wave4": ("CHARLS2018r.zip",),
    "2020-wave5": ("CHARLS2020r.zip",),
}


EXPECTED_TABLES: tuple[ExpectedTable, ...] = (
    ExpectedTable("2011", "基本信息", "demographic_background.dta", "direct"),
    ExpectedTable("2011", "家户登记表", "household_roster.dta", "direct"),
    ExpectedTable("2011", "家庭结构", "family_information.dta", "direct"),
    ExpectedTable("2011", "家庭交往及经济帮助", "family_transfer.dta", "direct"),
    ExpectedTable("2011", "健康状况与功能", "health_status_and_functioning.dta", "direct"),
    ExpectedTable("2011", "医疗保健与保险", "health_care_and_insurance.dta", "direct"),
    ExpectedTable("2011", "工作、退休、养老金", "work_retirement_and_pension.dta", "direct"),
    ExpectedTable("2011", "家庭收入、支出与资产", "household_income.dta", "direct"),
    ExpectedTable("2011", "个人收入、支出与资产", "individual_income.dta", "direct"),
    ExpectedTable("2011", "访员观察", "interviewer_observation.dta", "direct"),
    ExpectedTable("2011", "住房信息", "housing_characteristics.dta", "direct"),
    ExpectedTable("2011", "体检信息", "biomarkers.dta", "direct"),
    ExpectedTable("2011", "PSU 编码", "PSU.dta", "direct"),
    ExpectedTable("2011", "权重", "weight.dta", "direct"),
    ExpectedTable("2011", "社区问卷数据", "community.dta", "direct"),
    ExpectedTable("2011", "血检数据", "Blood_20140429.dta", "direct"),
    ExpectedTable("2011", "构建的子女数据库", "child.dta", "direct"),
    ExpectedTable("2011", "构建的父母数据库", "parent.dta", "direct"),
    ExpectedTable("2011", "构建的家户成员数据库", "hhmember.dta", "direct"),
    ExpectedTable("2011", "构建的支出、收入和财富数据库", "exp_income_wealth.dta", "direct"),
    ExpectedTable("2013", "基本信息", "Demographic_Background.dta", "bundle"),
    ExpectedTable("2013", "家庭信息", "Family_Information.dta", "bundle"),
    ExpectedTable("2013", "父母信息", "Parent.dta", "bundle"),
    ExpectedTable("2013", "子女信息", "Child.dta", "bundle"),
    ExpectedTable("2013", "其他家户成员信息", "Other_HHmember.dta", "bundle"),
    ExpectedTable("2013", "家庭经济交往", "Family_Transfer.dta", "bundle"),
    ExpectedTable("2013", "健康状况与功能", "Health_Status_and_Functioning.dta", "bundle"),
    ExpectedTable("2013", "医疗保健与保险", "Health_Care_and_Insurance.dta", "bundle"),
    ExpectedTable("2013", "工作退休及养老金", "Work_Retirement_and_Pension.dta", "bundle"),
    ExpectedTable("2013", "家户收入、支出及资产", "Household_Income.dta", "bundle"),
    ExpectedTable("2013", "个人收入及资产", "Individual_Income.dta", "bundle"),
    ExpectedTable("2013", "访问员观察", "Interviewer_Observation.dta", "bundle"),
    ExpectedTable("2013", "住房信息", "Housing_Characteristics.dta", "bundle"),
    ExpectedTable("2013", "体检信息", "Biomarker.dta", "bundle"),
    ExpectedTable("2013", "退出调查", "Exit_Interview.dta", "bundle"),
    ExpectedTable("2013", "死因信息", "Verbal_Autopsy.dta", "bundle"),
    ExpectedTable("2013", "PSU 编码", "PSU.dta", "bundle"),
    ExpectedTable("2013", "权重", "Weights.dta", "bundle"),
    ExpectedTable("2013", "构建的支出、收入和财富数据库", "exp_income_wealth.dta", "direct"),
    ExpectedTable("2015", "基本信息", "Demographic_Background.dta", "bundle"),
    ExpectedTable("2015", "家庭信息", "Family_Information.dta", "bundle"),
    ExpectedTable("2015", "家庭经济交往", "Family_Transfer.dta", "bundle"),
    ExpectedTable("2015", "健康状况与功能", "Health_Status_and_Functioning.dta", "bundle"),
    ExpectedTable("2015", "医疗保险与保健", "Health_Care_and_Insurance.dta", "bundle"),
    ExpectedTable("2015", "工作退休及养老金", "Work_Retirement_and_Pension.dta", "bundle"),
    ExpectedTable("2015", "家户收入、支出及资产", "Household_Income.dta", "bundle"),
    ExpectedTable("2015", "个人收入及资产", "Individual_Income.dta", "bundle"),
    ExpectedTable("2015", "住房信息", "Housing_Characteristics.dta", "bundle"),
    ExpectedTable("2015", "体检信息", "Biomarker.dta", "bundle"),
    ExpectedTable("2015", "血检数据", "Blood.dta", "direct"),
    ExpectedTable("2015", "样本权重", "Weights.dta", "bundle"),
    ExpectedTable("2015", "样本信息", "Sample_Infor.dta", "bundle"),
    ExpectedTable("2015", "构建的家户成员数据集", "Household_Member.dta", "bundle"),
    ExpectedTable("2015", "构建的父母数据集", "Parent.dta", "bundle"),
    ExpectedTable("2015", "构建的子女数据集", "Child.dta", "bundle"),
    ExpectedTable("2015", "构建的兄弟姐妹数据集", "Sibling.dta", "bundle"),
    ExpectedTable("2015", "构建的配偶的兄弟姐妹数据集", "Spousal_Sibling.dta", "bundle"),
    ExpectedTable("2018", "基本信息", "Demographic_Background.dta", "bundle"),
    ExpectedTable("2018", "家庭信息", "Family_Information.dta", "bundle"),
    ExpectedTable("2018", "家庭经济交往", "Family_Transfer.dta", "bundle"),
    ExpectedTable("2018", "健康状况与功能", "Health_Status_and_Functioning.dta", "bundle"),
    ExpectedTable("2018", "认知和抑郁", "Cognition.dta", "bundle"),
    ExpectedTable("2018", "知情人信息收集", "Insider.dta", "bundle"),
    ExpectedTable("2018", "医疗保健与保险", "Health_Care_and_Insurance.dta", "bundle"),
    ExpectedTable("2018", "工作和退休", "Work_Retirement.dta", "bundle"),
    ExpectedTable("2018", "养老金", "Pension.dta", "bundle"),
    ExpectedTable("2018", "家户收入、支出及资产", "Household_Income.dta", "bundle"),
    ExpectedTable("2018", "个人收入及资产", "Individual_Income.dta", "bundle"),
    ExpectedTable("2018", "房产和住房情况", "Housing.dta", "bundle"),
    ExpectedTable("2018", "样本权重", "Weights.dta", "bundle"),
    ExpectedTable("2018", "样本信息", "Sample_Infor.dta", "bundle"),
    ExpectedTable("2020", "基本信息", "Demographic_Background.dta", "bundle"),
    ExpectedTable("2020", "家庭信息", "Family_Information.dta", "bundle"),
    ExpectedTable("2020", "健康状况与功能", "Health_Status_and_Functioning.dta", "bundle"),
    ExpectedTable("2020", "工作和退休", "Work_Retirement.dta", "bundle"),
    ExpectedTable("2020", "家户收入与支出", "Household_Income.dta", "bundle"),
    ExpectedTable("2020", "个人收入", "Individual_Income.dta", "bundle"),
    ExpectedTable("2020", "疫情模块", "COVID_Module.dta", "bundle"),
    ExpectedTable("2020", "退出问卷", "Exit_Module.dta", "bundle"),
    ExpectedTable("2020", "样本信息", "Sample_Infor.dta", "bundle"),
    ExpectedTable("2020", "样本权重", "Weights.dta", "bundle"),
)


CHECKLIST_YEAR_RE = re.compile(r"^##\s+(?P<year>\d{4})")
CHECKLIST_SECTION_RE = re.compile(r"^###\s+(?P<section>.+)$")
CHECKLIST_ITEM_RE = re.compile(r"^-\s+(?P<item>.+?)(?:（(?P<detail>.+?)）)?$")
CHECKLIST_SHA_RE = re.compile(r"SHA1:\s*(?P<sha1>[0-9A-Fa-f]{40})")


def compute_sha1(path: Path) -> str:
    """计算单个文件的 SHA1。

    输入：
    - path: 要读取的文件路径。

    输出：
    - 40 位小写 SHA1 字符串。

    核心逻辑：
    - 按 1MB 分块读取，避免大文件一次性进入内存。
    """

    digest = sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checklist(path: Path = CHECKLIST_PATH) -> list[ChecklistEntry]:
    """解析官网对照清单 Markdown。

    输入：
    - path: `对照清单.md` 路径。

    输出：
    - 每个列表项对应一个 `ChecklistEntry`，包含年份、分区、条目名和可选 SHA1。

    说明：
    - 2011 年很多官网条目没有 SHA1，因此 `sha1_hex` 允许为空。
    - 清单文字是验收真源，脚本不改写条目名，只做轻量解析。
    """

    current_year = ""
    current_section = ""
    entries: list[ChecklistEntry] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        year_match = CHECKLIST_YEAR_RE.match(raw_line)
        if year_match:
            current_year = year_match.group("year")
            continue

        section_match = CHECKLIST_SECTION_RE.match(raw_line)
        if section_match:
            current_section = section_match.group("section").strip()
            continue

        item_match = CHECKLIST_ITEM_RE.match(raw_line)
        if not item_match or not current_year or not current_section:
            continue

        detail = item_match.group("detail") or ""
        sha_match = CHECKLIST_SHA_RE.search(detail)
        entries.append(
            ChecklistEntry(
                year=current_year,
                section=current_section,
                item=item_match.group("item").strip(),
                sha1_hex=sha_match.group("sha1").lower() if sha_match else None,
            )
        )
    return entries


def iter_files(root: Path) -> list[Path]:
    """递归列出目录中的普通文件，并按路径排序。"""

    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file()],
        key=lambda item: item.as_posix().lower(),
    )


def clean_directory(root: Path) -> None:
    """清空目录内容，但保留目录本身。

    这里专门用于清理 `data/extracted` 和 `data/curated`。
    RAW 原始文件不会调用这个函数，避免误删下载真源。
    """

    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def list_archive_members(archive: Path) -> list[str]:
    """使用系统 `tar` 列出 zip/rar 内部文件。

    Windows 这里的 `tar.exe` 来自系统 bsdtar，当前已验证能读本项目的 zip/rar。
    """

    proc = subprocess.run(
        ["tar", "-tf", str(archive)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"无法列出压缩包 {archive}: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def extract_archive(archive: Path, target_dir: Path) -> None:
    """把单个压缩包解压到指定目录。"""

    target_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(target_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"解压失败 {archive}: {proc.stderr.strip()}")


def rebuild_extracted(raw_root: Path = RAW_ROOT, extracted_root: Path = EXTRACTED_ROOT) -> list[dict[str, Any]]:
    """按 `EXTRACTION_PLAN` 重建 extracted 层。

    输出：
    - 每个解压包的文件名、SHA1、内部成员和目标波次。
    """

    clean_directory(extracted_root)
    records: list[dict[str, Any]] = []
    for wave in WAVES:
        wave_raw = raw_root / wave.slug
        wave_out = extracted_root / wave.slug
        wave_out.mkdir(parents=True, exist_ok=True)
        for archive_name in EXTRACTION_PLAN[wave.slug]:
            archive_path = wave_raw / archive_name
            if not archive_path.exists():
                records.append(
                    {
                        "wave": wave.slug,
                        "archive": archive_name,
                        "status": "missing",
                        "message": "EXTRACTION_PLAN 中列出，但 RAW 目录不存在。",
                    }
                )
                continue
            members = list_archive_members(archive_path)
            extract_archive(archive_path, wave_out)
            records.append(
                {
                    "wave": wave.slug,
                    "archive": archive_name,
                    "status": "ok",
                    "sha1": compute_sha1(archive_path),
                    "members": members,
                    "member_count": len(members),
                }
            )
    return records


def make_json_safe(value: Any) -> Any:
    """把 pyreadstat/pandas/numpy 对象转换成 JSON 可写结构。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def metadata_to_dict(dta_path: Path, frame: pd.DataFrame, meta: Any) -> dict[str, Any]:
    """把 pyreadstat 元数据整理成稳定 JSON。

    输入：
    - dta_path: 原始 Stata 文件。
    - frame: 已读取的数据表。
    - meta: pyreadstat 返回的 metadata 对象。

    输出：
    - 包含文件标签、变量标签、值标签、原始类型和缺失值信息的字典。
    """

    return make_json_safe(
        {
            "source_file": dta_path.as_posix(),
            "source_sha1": compute_sha1(dta_path),
            "source_backend": "pyreadstat",
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "columns": list(frame.columns),
            "file_label": getattr(meta, "file_label", None),
            "file_encoding": getattr(meta, "file_encoding", None),
            "notes": getattr(meta, "notes", None),
            "column_labels": getattr(meta, "column_labels", None),
            "column_names_to_labels": getattr(meta, "column_names_to_labels", None),
            "value_labels": getattr(meta, "value_labels", None),
            "variable_to_label": getattr(meta, "variable_to_label", None),
            "variable_value_labels": getattr(meta, "variable_value_labels", None),
            "original_variable_types": getattr(meta, "original_variable_types", None),
            "readstat_variable_types": getattr(meta, "readstat_variable_types", None),
            "missing_ranges": getattr(meta, "missing_ranges", None),
            "missing_user_values": getattr(meta, "missing_user_values", None),
            "variable_alignment": getattr(meta, "variable_alignment", None),
            "variable_storage_width": getattr(meta, "variable_storage_width", None),
            "variable_display_width": getattr(meta, "variable_display_width", None),
            "variable_measure": getattr(meta, "variable_measure", None),
        }
    )


def make_parquet_safe_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """生成适合写入 Parquet 的 DataFrame。

    背景：
    - Stata 支持 `.a` 到 `.z` 这类用户自定义缺失值。
    - `pyreadstat.read_dta(..., user_missing=True)` 会尽量保留这些缺失码。
    - 某些列因此会同时出现数字和字符串缺失码，例如 `1, 2, "d", "r"`。
    - CSV 可以直接写这种混合值，但 Parquet 要求单列类型稳定。

    输出：
    - 第一个返回值是可写 Parquet 的副本。
    - 第二个返回值记录哪些 object 列被转成字符串，便于后续建模时知道这里发生过规范化。
    """

    safe_frame = frame.copy()
    converted_columns: list[str] = []
    for column in safe_frame.columns:
        if safe_frame[column].dtype != "object":
            continue
        safe_frame[column] = safe_frame[column].astype("string")
        converted_columns.append(str(column))
    return safe_frame, converted_columns


def convert_dta_file(dta_path: Path, extracted_root: Path = EXTRACTED_ROOT, curated_root: Path = CURATED_ROOT) -> dict[str, Any]:
    """把一个 `.dta` 文件导出成 CSV、Parquet 和 metadata JSON。

    转换策略：
    - `apply_value_formats=False` 保留原始编码值，不把数值编码替换成文字标签。
    - `user_missing=True` 尽量保留 Stata 用户自定义缺失值，并把定义写进 metadata。
    - CSV 给人工查看，Parquet 给建模读取，metadata 保存标签语义。
    """

    relative = dta_path.relative_to(extracted_root)
    table_dir = curated_root / relative.parent / relative.stem
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{relative.stem}.csv"
    parquet_path = table_dir / f"{relative.stem}.parquet"
    metadata_path = table_dir / f"{relative.stem}.metadata.json"

    frame, meta = pyreadstat.read_dta(
        str(dta_path),
        apply_value_formats=False,
        user_missing=True,
    )
    parquet_frame, parquet_string_columns = make_parquet_safe_frame(frame)
    metadata = metadata_to_dict(dta_path, frame, meta)
    metadata.update(
        {
            "csv_file": csv_path.as_posix(),
            "parquet_file": parquet_path.as_posix(),
            "metadata_file": metadata_path.as_posix(),
            "curation_note": "CSV/Parquet 保留原始编码值；变量标签和值标签在 metadata JSON 中。",
            "parquet_string_columns": parquet_string_columns,
            "parquet_note": (
                "Parquet 要求单列类型稳定；含 Stata 特殊缺失码的 object 列会转成字符串，"
                "原始缺失码定义仍记录在 missing_user_values / value_labels 中。"
            ),
        }
    )

    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_frame.to_parquet(parquet_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "source": dta_path.as_posix(),
        "csv": csv_path.as_posix(),
        "parquet": parquet_path.as_posix(),
        "metadata": metadata_path.as_posix(),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "variable_label_count": len([label for label in metadata.get("column_labels", []) if label]),
        "value_label_set_count": len(metadata.get("value_labels", {}) or {}),
        "status": "ok",
    }


def rebuild_curated(extracted_root: Path = EXTRACTED_ROOT, curated_root: Path = CURATED_ROOT) -> list[dict[str, Any]]:
    """重建 curated 层，返回每张表的导出结果。"""

    clean_directory(curated_root)
    results: list[dict[str, Any]] = []
    for dta_path in sorted(extracted_root.rglob("*.dta"), key=lambda item: item.as_posix().lower()):
        try:
            results.append(convert_dta_file(dta_path, extracted_root, curated_root))
        except Exception as exc:  # noqa: BLE001
            error_path = curated_root / dta_path.relative_to(extracted_root).with_suffix(".error.txt")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                f"source={dta_path.as_posix()}\nerror={type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "source": dta_path.as_posix(),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_file": error_path.as_posix(),
                }
            )
    return results


def build_raw_file_index(raw_root: Path = RAW_ROOT) -> list[dict[str, Any]]:
    """扫描 RAW 文件，记录路径、大小、SHA1 和压缩包成员。"""

    records: list[dict[str, Any]] = []
    for path in iter_files(raw_root):
        if path.name == ".gitkeep":
            continue
        record: dict[str, Any] = {
            "path": path.as_posix(),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha1": compute_sha1(path),
            "kind": path.suffix.lower().lstrip(".") or "unknown",
        }
        if path.suffix.lower() in {".zip", ".rar"}:
            try:
                record["archive_members"] = list_archive_members(path)
                record["archive_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                record["archive_members"] = []
                record["archive_status"] = "error"
                record["archive_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def audit_checklist_coverage(
    checklist_entries: Sequence[ChecklistEntry],
    raw_files: Sequence[dict[str, Any]],
    extracted_root: Path = EXTRACTED_ROOT,
) -> list[dict[str, Any]]:
    """把官网清单条目映射到本地 RAW 和 extracted 产物。

    判定规则：
    - 有 SHA1 且本地文件直接命中：`sha1_match`。
    - 数据条目由总包解压出的 DTA 覆盖：`covered_by_extracted_dta`。
    - 2015 血检这种有候选 DTA 但 RAW SHA 不一致：`candidate_dta_but_sha_mismatch`。
    - 没有 SHA1 的 2011 条目，只能按文件/表名做存在性验收。
    """

    sha_to_files: dict[str, list[dict[str, Any]]] = {}
    for record in raw_files:
        sha_to_files.setdefault(str(record["sha1"]).lower(), []).append(record)

    expected_by_key = {(item.year, item.item): item for item in EXPECTED_TABLES}
    dta_by_wave: dict[str, set[str]] = {}
    for wave in WAVES:
        wave_dir = extracted_root / wave.slug
        dta_by_wave[wave.year] = {path.name for path in wave_dir.rglob("*.dta")}

    coverage: list[dict[str, Any]] = []
    for entry in checklist_entries:
        direct_matches = sha_to_files.get(entry.sha1_hex or "", [])
        expected_table = expected_by_key.get((entry.year, entry.item))
        dta_present = bool(expected_table and expected_table.dta_name in dta_by_wave.get(entry.year, set()))

        if direct_matches:
            status = "sha1_match"
        elif expected_table and dta_present and expected_table.source == "bundle":
            status = "covered_by_extracted_dta"
        elif expected_table and dta_present:
            status = "candidate_dta_but_sha_mismatch" if entry.sha1_hex else "dta_present_no_sha"
        elif entry.sha1_hex:
            status = "missing_sha1"
        else:
            status = "manual_review_no_sha"

        coverage.append(
            {
                "year": entry.year,
                "section": entry.section,
                "item": entry.item,
                "expected_sha1": entry.sha1_hex,
                "status": status,
                "matched_raw_files": [match["path"] for match in direct_matches],
                "expected_dta": expected_table.dta_name if expected_table else None,
                "expected_source": expected_table.source if expected_table else None,
                "dta_present": dta_present,
            }
        )
    return coverage


def summarize_by_wave(curated_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """按波次汇总 curated 输出。"""

    summary: dict[str, Any] = {}
    for result in curated_results:
        parts = Path(result["source"]).parts
        wave = parts[2] if len(parts) > 2 else "unknown"
        bucket = summary.setdefault(
            wave,
            {
                "tables": 0,
                "rows": 0,
                "columns": 0,
                "errors": 0,
                "variable_labels": 0,
                "value_label_sets": 0,
            },
        )
        if result.get("status") != "ok":
            bucket["errors"] += 1
            continue
        bucket["tables"] += 1
        bucket["rows"] += int(result.get("rows", 0))
        bucket["columns"] += int(result.get("columns", 0))
        bucket["variable_labels"] += int(result.get("variable_label_count", 0))
        bucket["value_label_sets"] += int(result.get("value_label_set_count", 0))
    return summary


def write_audit_json(
    raw_files: Sequence[dict[str, Any]],
    extraction_results: Sequence[dict[str, Any]],
    curated_results: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    audit_root: Path = AUDIT_ROOT,
) -> Path:
    """写出机器可读验收摘要。"""

    audit_root.mkdir(parents=True, exist_ok=True)
    path = audit_root / "charls_data_audit.json"
    payload = {
        "policy": {
            "checklist_source": CHECKLIST_PATH.as_posix(),
            "raw_root": RAW_ROOT.as_posix(),
            "extracted_root": EXTRACTED_ROOT.as_posix(),
            "curated_root": CURATED_ROOT.as_posix(),
            "note": "对照清单是验收真源；总包覆盖的单项数据以 extracted DTA 存在为准。",
        },
        "raw_files": list(raw_files),
        "extraction_results": list(extraction_results),
        "curated_results": list(curated_results),
        "coverage": list(coverage),
        "curated_summary_by_wave": summarize_by_wave(curated_results),
        "coverage_summary": {
            "total": len(coverage),
            "sha1_match": sum(1 for item in coverage if item["status"] == "sha1_match"),
            "covered_by_extracted_dta": sum(1 for item in coverage if item["status"] == "covered_by_extracted_dta"),
            "candidate_dta_but_sha_mismatch": sum(
                1 for item in coverage if item["status"] == "candidate_dta_but_sha_mismatch"
            ),
            "missing_sha1": sum(1 for item in coverage if item["status"] == "missing_sha1"),
            "manual_review_no_sha": sum(1 for item in coverage if item["status"] == "manual_review_no_sha"),
        },
    }
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_pipeline(*, skip_extract: bool = False, skip_curate: bool = False) -> dict[str, Any]:
    """执行完整整理流程。"""

    checklist_entries = parse_checklist()
    extraction_results: list[dict[str, Any]] = []
    curated_results: list[dict[str, Any]] = []

    if not skip_extract:
        extraction_results = rebuild_extracted()
    if not skip_curate:
        curated_results = rebuild_curated()

    raw_files = build_raw_file_index()
    coverage = audit_checklist_coverage(checklist_entries, raw_files)
    audit_path = write_audit_json(raw_files, extraction_results, curated_results, coverage)
    return {
        "audit_path": audit_path,
        "raw_files": raw_files,
        "extraction_results": extraction_results,
        "curated_results": curated_results,
        "coverage": coverage,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="整理 CHARLS RAW/extracted/curated 数据资产。")
    parser.add_argument("--skip-extract", action="store_true", help="跳过解压，只做后续步骤。")
    parser.add_argument("--skip-curate", action="store_true", help="跳过 curated 导出。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = build_argument_parser().parse_args(argv)
    result = run_pipeline(skip_extract=args.skip_extract, skip_curate=args.skip_curate)
    coverage = result["coverage"]
    curated_results = result["curated_results"]
    bad_coverage = [item for item in coverage if item["status"] in {"missing_sha1", "candidate_dta_but_sha_mismatch"}]
    bad_curated = [item for item in curated_results if item.get("status") != "ok"]

    print(f"已写出审计文件：{result['audit_path']}")
    print(f"清单条目：{len(coverage)}；需复核条目：{len(bad_coverage)}")
    print(f"curated 表：{sum(1 for item in curated_results if item.get('status') == 'ok')}；导出错误：{len(bad_curated)}")
    if bad_coverage:
        print("需复核的清单条目：")
        for item in bad_coverage:
            print(f"  - {item['year']} | {item['item']} | {item['status']}")
    if bad_curated:
        print("curated 导出错误：")
        for item in bad_curated:
            print(f"  - {item['source']} | {item.get('error')}")
    return 0 if not bad_curated else 1


if __name__ == "__main__":
    raise SystemExit(main())
