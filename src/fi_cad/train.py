"""命令行入口：训练 FI-CAD 第一版模型。"""

from __future__ import annotations

import argparse

from .config import load_config
from .modeling import train_all_models


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 2011 基线预测后续心脏病相关事件模型。")
    parser.add_argument("--config", default="configs/modeling.yaml", help="建模配置 YAML 路径。")
    return parser


def main() -> int:
    """训练所有配置模型。"""

    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    manifest = train_all_models(config)
    print(f"训练完成：{manifest['run_dir']}")
    print(f"样本数：{manifest['sample_count']}；阳性率：{manifest['positive_rate']:.4f}")
    print(f"指标表：{manifest['tables']['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
