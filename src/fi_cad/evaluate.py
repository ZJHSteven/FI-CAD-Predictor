"""命令行入口：汇总一次训练 run 的评估结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config


def resolve_run_dir(run: str, config: dict) -> Path:
    """解析 run 参数。

    `latest` 会读取 `output/runs/latest.txt`；其他值按路径处理。
    """

    if run == "latest":
        latest_path = Path(config["run"]["output_root"]) / "latest.txt"
        if not latest_path.exists():
            raise FileNotFoundError("没有找到 latest run，请先运行训练命令。")
        return Path(latest_path.read_text(encoding="utf-8").strip())
    return Path(run)


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="评估并打印 FI-CAD 模型 run 指标。")
    parser.add_argument("--config", default="configs/modeling.yaml", help="建模配置 YAML 路径。")
    parser.add_argument("--run", default="latest", help="run 目录或 latest。")
    return parser


def main() -> int:
    """读取指标表并输出简洁摘要。"""

    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(args.run, config)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(manifest["tables"]["metrics"])
    summary_path = run_dir / "reports" / "evaluation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Run：{run_dir}")
    print(metrics.to_string(index=False))
    print(f"评估摘要：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
