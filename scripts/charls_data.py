"""CHARLS 原始数据整理工具。

这个模块负责四件事：
1. 把原始下载目录从 `2011/2013/...` 规范化成 `2011-wave1` 这种命名。
2. 扫描每个波次的原始文件，记录压缩包状态、文件 SHA1 和异常。
3. 调用 7-Zip 解压 `.zip` / `.rar`，把结果放到 `data/extracted`。
4. 把解压后的 `.dta` 导出成 CSV，并额外保存元数据 JSON，尽量减少格式损失。

说明：
- 这里不把 CSV 当成唯一真源，Stata 原件和元数据必须同时保留。
- 若某个压缩包本身损坏或为空，脚本不会静默吞掉，而是明确写进清单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
import argparse
from collections.abc import Mapping, Sequence as SequenceABC
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Sequence

import pandas as pd

try:
    import pyreadstat
except ImportError:  # pragma: no cover - 环境里没有这个包时自动回退 pandas。
    pyreadstat = None


@dataclass(frozen=True)
class WaveSpec:
    """描述一个 CHARLS 波次的最小信息。

    字段说明：
    - year: 这波对应的调查年份。
    - wave: 这波在正式追踪中的波次编号。
    - slug: 规范化目录名，比如 `2011-wave1`。
    - legacy_dir_name: 老目录名，比如 `2011`，用于自动迁移旧布局。
    """

    year: int
    wave: int
    slug: str
    legacy_dir_name: str

    @property
    def label(self) -> str:
        """返回给人看的波次标题。"""

        return f"{self.year} 年 wave{self.wave}"


WAVE_SPECS = [
    # 这里把年份和波次一起固定下来，后面清单、解压和转换都按这个顺序走。
    # 用户要求的命名是“年份 + waveX”，所以这里统一成 `2011-wave1` 这种形式。
    # 这样既能看出时间顺序，也能看出它属于第几波。
    # 先把顺序写死，避免以后扫描时波次乱序。
    # 这对清单和后续建模都很重要。
    # 这里不再引入别的年份，避免把 pilot 或 life history 混进主线。
    # 以后如果要扩展专题调查，再单独加表，不和主线混用。
    #
    # 2011 -> wave1
    # 2013 -> wave2
    # 2015 -> wave3
    # 2018 -> wave4
    # 2020 -> wave5
    #
    WaveSpec(2011, 1, "2011-wave1", "2011"),
    WaveSpec(2013, 2, "2013-wave2", "2013"),
    WaveSpec(2015, 3, "2015-wave3", "2015"),
    WaveSpec(2018, 4, "2018-wave4", "2018"),
    WaveSpec(2020, 5, "2020-wave5", "2020"),
]

ARCHIVE_EXTENSIONS = {".zip", ".rar"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".rtf"}
PARTIAL_SUFFIXES = {".part", ".partial"}
CHUNK_SIZE = 1024 * 1024

LISTING_ROW_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+\s+\S+\s+\S+\s+(?P<name>.+?)\s*$"
)
SLT_PATH_RE = re.compile(r"^Path = (?P<path>.+)$")
WARNING_RE = re.compile(r"WARNINGS?:\s*(\d+)", re.IGNORECASE)
ERROR_RE = re.compile(r"ERRORS?:\s*(\d+)", re.IGNORECASE)


@dataclass
class FileRecord:
    """单个文件的盘点结果。"""

    path: Path
    kind: str
    size_bytes: int
    sha1_hex: str
    status: str
    note: str
    inner_files: list[str] = field(default_factory=list)


@dataclass
class WaveInventory:
    """一个波次的原始数据盘点。"""

    spec: WaveSpec
    raw_dir: Path
    files: list[FileRecord]
    archive_ok: int
    archive_warn: int
    archive_error: int
    document_count: int
    partial_count: int


@dataclass
class ArchiveInspection:
    """压缩包的检测结果。"""

    path: Path
    status: str
    returncode: int
    warning_count: int
    error_count: int
    inner_files: list[str]
    note: str


@dataclass
class ConversionResult:
    """单个 `.dta` 文件的导出结果。"""

    source: Path
    csv_path: Path | None
    meta_path: Path | None
    row_count: int
    column_count: int
    status: str
    note: str


def compute_sha1(path: Path) -> str:
    """计算文件 SHA1。

    这里按流式读取，避免大文件一次性进内存。
    """

    digest = sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_7z_executable() -> Path:
    """寻找可用的 7-Zip 可执行文件。

    优先级：
    1. 环境变量 `SEVEN_ZIP_EXE`
    2. PATH 里的 `7z`
    3. 常见安装路径
    """

    env_path = os.environ.get("SEVEN_ZIP_EXE")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    which_path = shutil.which("7z")
    if which_path:
        candidates.append(Path(which_path))
    candidates.extend(
        [
            Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe"),
            Path(r"C:\Program Files\7-Zip\7z.exe"),
            Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "没有找到可用的 7z.exe。请设置 SEVEN_ZIP_EXE，或者安装 7-Zip。"
    )


def classify_file(path: Path) -> tuple[str, str]:
    """根据扩展名给文件做最粗粒度分类。

    返回值是 `(kind, note)`：
    - `kind` 用于清单表格
    - `note` 用于提示为什么被这样分类
    """

    suffix = path.suffix.lower()
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive", "可解压压缩包"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document", "说明文档或问卷"
    if suffix in PARTIAL_SUFFIXES:
        return "partial", "未完成下载或分卷残留"
    return "other", "未识别类型"


def normalize_raw_layout(data_root: Path) -> list[tuple[Path, Path]]:
    """把老的年份目录迁移成统一的 `year-waveX` 命名。

    返回值记录了所有实际发生的迁移，方便写日志或测试。
    """

    raw_root = data_root / "raw"
    moved: list[tuple[Path, Path]] = []
    for spec in WAVE_SPECS:
        legacy_dir = raw_root / spec.legacy_dir_name
        canonical_dir = raw_root / spec.slug
        if legacy_dir.exists() and canonical_dir.exists():
            if legacy_dir.resolve() != canonical_dir.resolve():
                raise FileExistsError(
                    f"旧目录 {legacy_dir} 和新目录 {canonical_dir} 同时存在，"
                    "脚本无法判断应该以哪一个为准。"
                )
            continue
        if legacy_dir.exists():
            legacy_dir.rename(canonical_dir)
            moved.append((legacy_dir, canonical_dir))
    return moved


def get_wave_dir(data_root: Path, spec: WaveSpec) -> Path:
    """找到某个波次的原始目录。

    如果新目录已经存在，就直接返回新目录。
    如果新目录不存在但老目录存在，就返回老目录。
    """

    raw_root = data_root / "raw"
    canonical = raw_root / spec.slug
    legacy = raw_root / spec.legacy_dir_name
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    return canonical


def iter_data_files(root: Path) -> list[Path]:
    """递归列出目录下的全部文件，排除隐藏说明文件。"""

    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"],
        key=lambda item: str(item).lower(),
    )


def inspect_archive(path: Path, seven_zip: Path) -> ArchiveInspection:
    """调用 7-Zip 读取压缩包的列表信息。

    这里我们不只看返回码，还看输出中的 `Warnings:` / `Errors:`。
    这是因为有些压缩包虽然带头部异常，但仍然可以列出并解压出核心文件。
    """

    proc = subprocess.run(
        [str(seven_zip), "l", "-slt", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    warning_count = _extract_counter(stdout, WARNING_RE)
    error_count = _extract_counter(stdout, ERROR_RE)
    inner_files = _extract_archive_listing(stdout)

    note_parts: list[str] = []
    if warning_count:
        note_parts.append(f"{warning_count} 个警告")
    if error_count:
        note_parts.append(f"{error_count} 个错误")
    if not note_parts and proc.returncode == 0:
        status = "ok"
        note_parts.append("结构正常")
    elif inner_files:
        status = "warn"
        note_parts.append("可列出文件，但压缩包头部不规范")
    else:
        status = "error"
        note_parts.append("无法作为正常压缩包读取")

    if stderr.strip():
        note_parts.append(stderr.strip().splitlines()[0])

    return ArchiveInspection(
        path=path,
        status=status,
        returncode=proc.returncode,
        warning_count=warning_count,
        error_count=error_count,
        inner_files=inner_files,
        note="；".join(note_parts),
    )


def extract_archive(path: Path, output_dir: Path, seven_zip: Path) -> tuple[bool, str]:
    """把单个压缩包解压到指定目录。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(seven_zip), "x", "-y", f"-o{str(output_dir)}", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    message_parts = []
    if stdout.strip():
        message_parts.append(stdout.strip().splitlines()[-1])
    if stderr.strip():
        message_parts.append(stderr.strip().splitlines()[0])

    extracted_count = sum(1 for item in output_dir.rglob("*") if item.is_file())
    if extracted_count > 0:
        if proc.returncode == 0:
            return True, "；".join(message_parts) if message_parts else "解压完成"
        # 7-Zip 对“有警告但仍能吐出文件”的档案可能返回非零。
        # 这类情况我们视作“可用但需要复核”，不要直接把整条流水线判死。
        warning_text = "；".join(message_parts) if message_parts else f"7z 返回码 {proc.returncode}"
        return True, f"解压完成但有异常：{warning_text}"
    if proc.returncode in (0, 1):
        return True, "；".join(message_parts) if message_parts else "解压完成"
    return False, "；".join(message_parts) if message_parts else f"7z 返回码 {proc.returncode}"


def convert_stata_tree(extracted_root: Path, curated_root: Path) -> list[ConversionResult]:
    """把 extracted 里的 `.dta` 统一导出成 CSV + 元数据 JSON。"""

    results: list[ConversionResult] = []
    for dta_path in sorted(extracted_root.rglob("*.dta")):
        relative = dta_path.relative_to(extracted_root)
        csv_path = curated_root / relative.with_suffix(".csv")
        meta_path = curated_root / relative.with_suffix(".meta.json")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            frame, metadata = read_stata_table(dta_path)
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
            metadata.update(
                {
                    "source_file": str(dta_path),
                    "csv_file": str(csv_path),
                    "row_count": int(frame.shape[0]),
                    "column_count": int(frame.shape[1]),
                    "columns": list(frame.columns),
                    "note": (
                        "CSV 便于阅读，但 Stata 的缺失码、标签和一些类型信息仍建议结合原始 .dta 使用。"
                    ),
                }
            )
            metadata = make_json_safe(metadata)
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(
                ConversionResult(
                    source=dta_path,
                    csv_path=csv_path,
                    meta_path=meta_path,
                    row_count=int(frame.shape[0]),
                    column_count=int(frame.shape[1]),
                    status="ok",
                    note="导出成功",
                )
            )
        except Exception as exc:  # noqa: BLE001
            error_path = curated_root / relative.with_suffix(".error.txt")
            error_path.write_text(
                "\n".join(
                    [
                        f"source={dta_path}",
                        f"error={type(exc).__name__}: {exc}",
                        "note=当前 Python Stata 读取后端无法稳定解析这个文件，保留原始 .dta 供后续人工处理。",
                    ]
                ),
                encoding="utf-8",
            )
            results.append(
                ConversionResult(
                    source=dta_path,
                    csv_path=None,
                    meta_path=error_path,
                    row_count=0,
                    column_count=0,
                    status="error",
                    note=str(exc),
                )
            )
    return results


def read_stata_table(dta_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """优先用 pyreadstat 读 Stata 文件，失败后再回退 pandas。

    这样做的原因是：
    - pyreadstat 对 Stata / SPSS / SAS 的兼容性通常更强；
    - pandas 在部分 `.dta` 上会直接报缓冲区错误；
    - 如果两者都失败，就让上层记录错误文本，而不是静默丢数据。
    """

    if pyreadstat is not None:
        try:
            frame, meta = pyreadstat.read_dta(str(dta_path))
            metadata = {
                "source_backend": "pyreadstat",
                "variable_labels": {
                    column: label
                    for column, label in zip(
                        list(getattr(meta, "column_names", [])),
                        list(getattr(meta, "column_labels", [])),
                    )
                },
                "value_labels": make_json_safe(getattr(meta, "value_labels", {})),
                "data_label": make_json_safe(getattr(meta, "file_label", None)),
            }
            return frame, metadata
        except Exception:
            # 先让 pandas 再试一次，避免 pyreadstat 个别版本/编码问题直接卡死。
            pass

    reader = pd.io.stata.StataReader(str(dta_path))
    frame = reader.read(convert_categoricals=True, preserve_dtypes=True)
    metadata = {
        "source_backend": "pandas",
        "variable_labels": make_json_safe(reader.variable_labels()),
        "value_labels": make_json_safe(reader.value_labels()),
        "data_label": make_json_safe(getattr(reader, "data_label", None)),
    }
    return frame, metadata


def make_json_safe(value: object) -> object:
    """把 pandas / numpy / Path 等对象转换成 JSON 可以接受的普通结构。

    CHARLS 的 Stata 元数据里经常会出现 `numpy.int32` 这类对象。
    如果不做这个转换，`json.dumps` 会直接报错，导致元数据文件写不出来。
    """

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for key, item in value.items():
            safe_mapping[str(make_json_safe(key))] = make_json_safe(item)
        return safe_mapping
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def scan_wave_inventory(spec: WaveSpec, data_root: Path, seven_zip: Path | None = None) -> WaveInventory:
    """扫描某个波次的原始文件，并给出可读的统计结果。"""

    raw_dir = get_wave_dir(data_root, spec)
    files = []
    archive_ok = 0
    archive_warn = 0
    archive_error = 0
    document_count = 0
    partial_count = 0
    file_paths = iter_data_files(raw_dir)
    for file_path in file_paths:
        kind, note = classify_file(file_path)
        sha1_hex = compute_sha1(file_path)
        status = "ok"
        inner_files: list[str] = []
        if kind == "archive":
            if seven_zip is None:
                seven_zip = find_7z_executable()
            inspection = inspect_archive(file_path, seven_zip)
            status = inspection.status
            inner_files = inspection.inner_files
            note = inspection.note
            if status == "ok":
                archive_ok += 1
            elif status == "warn":
                archive_warn += 1
            else:
                archive_error += 1
        elif kind == "document":
            document_count += 1
        elif kind == "partial":
            partial_count += 1
            status = "warn"
        files.append(
            FileRecord(
                path=file_path,
                kind=kind,
                size_bytes=file_path.stat().st_size,
                sha1_hex=sha1_hex,
                status=status,
                note=note,
                inner_files=inner_files,
            )
        )
    return WaveInventory(
        spec=spec,
        raw_dir=raw_dir,
        files=files,
        archive_ok=archive_ok,
        archive_warn=archive_warn,
        archive_error=archive_error,
        document_count=document_count,
        partial_count=partial_count,
    )


def build_manifest(data_root: Path, inventories: Sequence[WaveInventory], extraction_root: Path, curated_root: Path) -> str:
    """把扫描结果渲染成 Markdown 清单。"""

    lines: list[str] = []
    lines.append("# CHARLS 数据清单")
    lines.append("")
    lines.append("## 目录约定")
    lines.append("- `data/raw/<year-waveX>`: 原始下载文件，保留压缩包和原始文档。")
    lines.append("- `data/extracted/<year-waveX>`: 把原始压缩包解出来后的 `.dta` 原件。")
    lines.append("- `data/curated/<year-waveX>`: 从 `.dta` 导出的 CSV 和元数据 JSON。")
    lines.append("- 目录命名统一使用 `2011-wave1` 这种形式，保留年份和波次信息。")
    lines.append("")
    lines.append("## 当前状态总览")
    lines.append(
        "| 波次 | 原始目录 | 文件数 | 文档 | 压缩包正常 | 压缩包警告 | 压缩包错误 | 已解压 DTA | CSV | 元数据 | 错误报告 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for inv in inventories:
        extracted_wave_root = extraction_root / inv.spec.slug
        curated_wave_root = curated_root / inv.spec.slug
        extracted_dta_count = sum(1 for _ in extracted_wave_root.rglob("*.dta")) if extracted_wave_root.exists() else 0
        curated_csv_count = sum(1 for _ in curated_wave_root.rglob("*.csv")) if curated_wave_root.exists() else 0
        curated_meta_count = sum(1 for _ in curated_wave_root.rglob("*.meta.json")) if curated_wave_root.exists() else 0
        curated_error_count = sum(1 for _ in curated_wave_root.rglob("*.error.txt")) if curated_wave_root.exists() else 0
        lines.append(
            f"| {inv.spec.label} | `{inv.raw_dir.as_posix()}` | {len(inv.files)} | "
            f"{inv.document_count} | {inv.archive_ok} | {inv.archive_warn} | {inv.archive_error} | "
            f"{extracted_dta_count} | {curated_csv_count} | {curated_meta_count} | {curated_error_count} |"
        )
    lines.append("")
    lines.append("## 盘点说明")
    lines.append(
        "- 下面的表格记录的是当前工作区里已经落地的原始文件，而不是官网所有理论上存在的条目。"
    )
    lines.append(
        "- 如果某个压缩包能列出内部文件但 7-Zip 报了头部错误，会标成 `warn`；如果完全打不开，会标成 `error`。"
    )
    lines.append(
        "- CSV 导出会保留数值和字符串内容，但 Stata 的标签和一些细节类型仍建议结合 `.dta` 原件和元数据 JSON。"
    )
    lines.append("")

    for inv in inventories:
        lines.append(f"## {inv.spec.label}")
        lines.append("")
        lines.append(f"- 原始目录：`{inv.raw_dir.as_posix()}`")
        lines.append(
            f"- 解压目录：`{(extraction_root / inv.spec.slug).as_posix()}`"
        )
        lines.append(
            f"- 可读化目录：`{(curated_root / inv.spec.slug).as_posix()}`"
        )
        lines.append(
            f"- 统计：{len(inv.files)} 个文件，{inv.document_count} 个文档，"
            f"{inv.archive_ok} 个正常压缩包，{inv.archive_warn} 个带警告压缩包，"
            f"{inv.archive_error} 个错误压缩包，{inv.partial_count} 个疑似未完成下载。"
        )
        extracted_wave_root = extraction_root / inv.spec.slug
        curated_wave_root = curated_root / inv.spec.slug
        extracted_dta_count = sum(1 for _ in extracted_wave_root.rglob("*.dta")) if extracted_wave_root.exists() else 0
        curated_csv_count = sum(1 for _ in curated_wave_root.rglob("*.csv")) if curated_wave_root.exists() else 0
        curated_meta_count = sum(1 for _ in curated_wave_root.rglob("*.meta.json")) if curated_wave_root.exists() else 0
        curated_error_count = sum(1 for _ in curated_wave_root.rglob("*.error.txt")) if curated_wave_root.exists() else 0
        lines.append(
            f"- 解压与导出：{extracted_dta_count} 个 `.dta`，{curated_csv_count} 个 CSV，"
            f"{curated_meta_count} 个元数据 JSON，{curated_error_count} 个错误报告。"
        )
        lines.append("")
        lines.append("| 文件 | 类型 | 大小(B) | SHA1 | 状态 | 内部文件 / 备注 |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        for record in inv.files:
            inner = "；".join(record.inner_files) if record.inner_files else record.note
            lines.append(
                f"| `{record.path.name}` | {record.kind} | {record.size_bytes} | "
                f"`{record.sha1_hex}` | {record.status} | {inner} |"
            )
        lines.append("")

    lines.append("## 后续落地目录")
    lines.append("- `data/extracted`：放解压后的原始 Stata 文件。")
    lines.append("- `data/curated`：放 CSV 和元数据 JSON，后面建模优先读这里。")
    lines.append("- 如果后续再补下载，只要重新跑整理脚本即可自动刷新清单。")
    lines.append("")
    return "\n".join(lines)


def write_manifest(data_root: Path, inventories: Sequence[WaveInventory]) -> Path:
    """写出 `data/MANIFEST.md`。"""

    extraction_root = data_root / "extracted"
    curated_root = data_root / "curated"
    manifest_path = data_root / "MANIFEST.md"
    manifest = build_manifest(data_root, inventories, extraction_root, curated_root)
    manifest_path.write_text(manifest, encoding="utf-8")
    return manifest_path


def ensure_workspace_dirs(data_root: Path) -> None:
    """确保数据目录骨架存在。"""

    for relative in [
        "raw",
        "extracted",
        "curated",
    ]:
        (data_root / relative).mkdir(parents=True, exist_ok=True)


def run_pipeline(data_root: Path, *, skip_extract: bool = False, skip_convert: bool = False) -> dict[str, object]:
    """执行完整的数据整理流程。

    返回一个字典，便于脚本层打印汇总信息。
    """

    ensure_workspace_dirs(data_root)
    moved = normalize_raw_layout(data_root)
    seven_zip = find_7z_executable()
    inventories = [scan_wave_inventory(spec, data_root, seven_zip) for spec in WAVE_SPECS]

    extraction_root = data_root / "extracted"
    curated_root = data_root / "curated"
    extraction_results: list[dict[str, object]] = []
    conversion_results: list[ConversionResult] = []

    if not skip_extract:
        for inv in inventories:
            wave_extract_root = extraction_root / inv.spec.slug
            wave_extract_root.mkdir(parents=True, exist_ok=True)
            for record in inv.files:
                if record.kind != "archive":
                    continue
                archive_rel = record.path.relative_to(inv.raw_dir)
                target_dir = wave_extract_root / archive_rel.with_suffix("")
                ok, message = extract_archive(record.path, target_dir, seven_zip)
                extraction_results.append(
                    {
                        "archive": str(record.path),
                        "target_dir": str(target_dir),
                        "ok": ok,
                        "message": message,
                    }
                )

    if not skip_convert:
        conversion_results = convert_stata_tree(extraction_root, curated_root)

    inventories = [scan_wave_inventory(spec, data_root, seven_zip) for spec in WAVE_SPECS]
    manifest_path = write_manifest(data_root, inventories)

    return {
        "moved": moved,
        "inventories": inventories,
        "extraction_results": extraction_results,
        "conversion_results": conversion_results,
        "manifest_path": manifest_path,
    }


def _extract_counter(text: str, pattern: re.Pattern[str]) -> int:
    match = pattern.search(text)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _extract_archive_listing(text: str) -> list[str]:
    """从 7-Zip 的表格输出里提取内部文件名。"""

    names: list[str] = []
    in_item_section = False
    for line in text.splitlines():
        if line.strip() == "----------":
            in_item_section = True
            continue
        if not in_item_section:
            continue
        slt_match = SLT_PATH_RE.match(line)
        if slt_match:
            path_text = slt_match.group("path").strip()
            if path_text and path_text not in names:
                names.append(path_text)
            continue
        match = LISTING_ROW_RE.match(line)
        if match:
            name = match.group("name").strip()
            if name and name not in names:
                names.append(name)
    return names


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="整理 CHARLS 原始数据，生成清单、解压产物和 CSV 导出。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="数据根目录，默认是仓库里的 data。",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="只生成清单，不做压缩包解压。",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="只做解压和清单，不把 .dta 导出成 CSV。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(
        args.data_root,
        skip_extract=args.skip_extract,
        skip_convert=args.skip_convert,
    )

    moved = result["moved"]
    if moved:
        print("已规范化目录命名：")
        for old_path, new_path in moved:
            print(f"  - {old_path} -> {new_path}")
    else:
        print("目录命名已经是规范状态，未发生迁移。")

    manifest_path = result["manifest_path"]
    print(f"已生成清单：{manifest_path}")

    extraction_results = result["extraction_results"]
    if extraction_results:
        print("压缩包解压摘要：")
        for item in extraction_results:
            state = "成功" if item["ok"] else "失败"
            print(f"  - {state} | {item['archive']} -> {item['target_dir']} | {item['message']}")
    else:
        print("本次没有执行解压。")

    conversion_results = result["conversion_results"]
    if conversion_results:
        ok_count = sum(1 for item in conversion_results if item.status == "ok")
        err_count = sum(1 for item in conversion_results if item.status != "ok")
        print(f"CSV 导出摘要：成功 {ok_count} 个，失败 {err_count} 个。")
    else:
        print("本次没有执行 CSV 导出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
