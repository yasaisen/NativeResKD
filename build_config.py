from __future__ import annotations

import argparse
from pathlib import Path

from NativeResKD.configs.NRKD_baseConfig import NRKD_baseConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resolved NativeResKD 4.01 project config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    cfg = NRKD_baseConfig(project_name="NativeResKD", is_HPC=args.hpc)
    written = cfg.save(str(target.parent), target.name)
    print(written)


if __name__ == "__main__":
    main()
