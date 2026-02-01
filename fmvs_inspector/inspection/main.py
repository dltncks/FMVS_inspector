# fmvs_inspector/inspection/main.py
"""
Entry point for running inspection.

Preferred run:
  python -m fmvs_inspector
"""
from fmvs_inspector.inspection.config import DEFAULT_CONFIG
from fmvs_inspector.inspection.run_inspection import run_inspection


def main() -> None:
    run_inspection(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
