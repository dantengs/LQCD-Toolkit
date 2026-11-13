"""Small command line interface; no remote access or database writes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import sqlite3

from .database import CorrelatorStore


def _json(value):
    return json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read and analyze the versioned correlator database")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List stored original paths")
    listing.add_argument("database")
    listing.add_argument("--root-id")
    listing.add_argument("--glob")
    listing.add_argument("--after-id", type=int, default=0)
    listing.add_argument("--limit", type=int, default=50)
    inspect = sub.add_parser("inspect", help="Inspect one original, optionally decode numeric arrays")
    inspect.add_argument("database")
    inspect.add_argument("--id", type=int, required=True)
    inspect.add_argument("--profile")
    verify = sub.add_parser("verify", help="Verify DB seal hash; optionally every original payload")
    verify.add_argument("database")
    verify.add_argument("--all-payloads", action="store_true")
    fit = sub.add_parser("fit2", help="Fit an explicit two-point selection")
    fit.add_argument("settings")
    fit.add_argument("--output", required=True)
    fit.add_argument("--verify-database-hash", action="store_true")
    joint = sub.add_parser("joint", help="Fit initial C2, final C2 and C3 from SQLite")
    joint.add_argument("settings")
    joint.add_argument("--output", required=True)
    joint.add_argument("--verify-database-hash", action="store_true")
    check = sub.add_parser("check", help="Check the physical workflow without starting a solver")
    check.add_argument("configuration")
    run = sub.add_parser("run", help="Run source, propagator solve, contractions and direct SQLite output")
    run.add_argument("configuration")
    args = parser.parse_args(argv)
    try:
        if args.command in {"check", "run"}:
            from .workflow import plan_workflow, run_workflow
            if args.command == "check":
                plan = plan_workflow(args.configuration)
                print(_json({"status": "configuration-checked-no-solver-started", "output": plan["output"],
                             "measurements": len(plan["tasks"]), "ensemble": plan["config"]["ensemble"],
                             "peak_reference_memory_bytes": max(t["estimated_memory_bytes"] for t in plan["tasks"]),
                             "scientific_mode": plan["config"]["scientific_mode"]}))
            else:
                print(_json(run_workflow(args.configuration)))
            return 0
        if args.command in {"fit2", "joint"}:
            if args.command == "fit2":
                from .analysis import run_fit_file
                result = run_fit_file(args.settings, verify_database_hash=args.verify_database_hash)
            else:
                from .joint_analysis import run_joint_file
                result = run_joint_file(args.settings, verify_database_hash=args.verify_database_hash)
            target = Path(args.output)
            with target.open("x", encoding="utf-8") as output:
                output.write(_json(result) + "\n")
            print(_json({"result": str(target.absolute()), "status": result["status"],
                         "parameters": result["parameters"], "standard_errors": result["standard_errors"],
                         "chi2": result["chi2"], "dof": result["dof"]}))
            return 0
        with CorrelatorStore(args.database, verify_database_hash=args.command == "verify") as store:
            if args.command == "list":
                result = {"database": store.summary(), "files": [asdict(r) for r in store.iter_files(
                    root_id=args.root_id, path_glob=args.glob, after_id=args.after_id, limit=args.limit)]}
            elif args.command == "inspect":
                record = store.file(args.id)
                payload = store.read_original(record.id)
                result = {"original": asdict(record), "payload_verified": True, "byte_count": len(payload)}
                if args.profile:
                    decoded = store.decode(record.id, profile=args.profile)
                    result["decode"] = {"profile": decoded.profile, "metadata": decoded.metadata,
                                        "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str}
                                                   for k, v in decoded.arrays.items()}}
            else:
                count, cursor = 0, 0
                if args.all_payloads:
                    while True:
                        records = list(store.iter_files(after_id=cursor, limit=1000))
                        if not records:
                            break
                        for record in records:
                            store.read_original(record.id)
                            count += 1
                        cursor = records[-1].id
                result = store.summary() | {"payloads_verified": count}
            print(_json(result))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"Analysis stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
