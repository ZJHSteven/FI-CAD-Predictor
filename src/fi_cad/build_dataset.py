"""命令行入口：构建第一版 FI-CAD 建模数据集。"""

from __future__ import annotations

import argparse

from .config import load_config
from .data import build_modeling_dataset, write_dataset_outputs


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="构建 2011 基线 -> 2020 心脏病相关事件建模数据集。")
    parser.add_argument("--config", default="configs/modeling.yaml", help="建模配置 YAML 路径。")
    return parser


def main() -> int:
    """执行数据集构建并打印关键摘要。"""

    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    result = build_modeling_dataset(config)
    paths = write_dataset_outputs(result, config)
    target = config["dataset"]["endpoint_name"]
    positive_rate = float(result.dataset[target].mean()) if len(result.dataset) else 0.0
    print(f"建模数据集：{paths['dataset']}")
    print(f"样本数：{len(result.dataset)}；阳性率：{positive_rate:.4f}")
    print(f"特征数：{len([c for c in result.dataset.columns if c not in {'ID', target}])}")
    print(f"变量字典：{paths['variable_dictionary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
