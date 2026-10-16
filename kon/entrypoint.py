"""Standalone application and finite, hardware-free verification commands."""
from __future__ import annotations

import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()
    if any(flag in sys.argv[1:] for flag in ("--package-check", "--speech-self-test")):
        from kon.package_check import main as check_main
        return check_main()
    from kon.app import main as app_main
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
