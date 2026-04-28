"""CHARLS 数据整理脚本入口。

这个文件只做两件事：
1. 把 `scripts` 目录加到 Python 路径里，方便直接运行仓库内代码。
2. 调用 `charls_data.main()`。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """把仓库的 `src` 目录放进 Python 搜索路径。"""

    script_root = Path(__file__).resolve().parent
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))


def main() -> int:
    """脚本入口。"""

    _bootstrap_src_path()
    from charls_data import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
