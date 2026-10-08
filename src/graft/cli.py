"""Command-line entry point: ``graft validate ...``."""

from __future__ import annotations

import argparse
import sys

from graft.io.luh import LUHScenario
from graft.validate import DEFAULT_TOL, check_scenario, format_report


def _cmd_validate(args: argparse.Namespace) -> int:
    scen = LUHScenario.from_paths(
        states=args.states,
        transitions=args.transitions,
        management=args.management,
        static=args.static,
    )
    try:
        report = check_scenario(
            scen,
            name=args.name,
            tol=args.tol,
            year_stride=args.stride,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
    finally:
        scen.close()
    print(format_report(report))
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="graft")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="conservation report on a LUH3 scenario")
    v.add_argument("--states", required=True)
    v.add_argument("--transitions", required=True)
    v.add_argument("--management", default=None)
    v.add_argument("--static", default=None)
    v.add_argument("--name", default="scenario")
    v.add_argument("--tol", type=float, default=DEFAULT_TOL)
    v.add_argument("--stride", type=int, default=1, help="check every Nth year-step")
    v.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=_cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
