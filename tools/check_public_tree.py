"""Check that the public tree contains only new code and anonymous test fixtures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "validation", "__pycache__", "build", "dist", ".venv"}
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Documents|CodeXFiles)[\\/]|/(?:home|users|scratch)/[A-Za-z0-9_.-]+/)", re.I)
SECRET = re.compile(r"(?:-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,})")


def main():
    problems, inspected, synthetic_binaries = [], 0, 0
    provenance_path = ROOT / "tests/fixtures/correlators/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("origin") != "synthetic-format-fixtures":
        problems.append(("tests/fixtures/correlators", "fixtures must be explicitly synthetic"))
    fixture_records = {item["local_name"]: item for item in provenance["files"]}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or set(relative.parts) & SKIP or any(x.endswith(".egg-info") for x in relative.parts):
            continue
        if path.is_symlink():
            problems.append((str(relative), "links must not hide unreviewed content"))
            continue
        if set(relative.parts) & {"legacy", "old_repository", "discovery", "private-backup", "archives"}:
            problems.append((str(relative), "historical or private content is outside this repository"))
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".7z", ".bundle", ".pyc", ".ipynb", ".sqlite"}:
            problems.append((str(relative), "archives, notebooks, bytecode and databases are not public source"))
            continue
        if relative.parts[:3] == ("tests", "fixtures", "correlators"):
            expected = fixture_records.get(path.name)
            if path.name != "provenance.json":
                data = path.read_bytes()
                if expected is None or len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    problems.append((str(relative), "synthetic fixture checksum mismatch"))
        if path.suffix == ".bin":
            if relative.parts[:3] != ("tests", "fixtures", "correlators"):
                problems.append((str(relative), "unapproved numeric binary"))
            else:
                synthetic_binaries += 1
            continue
        items = [(relative.as_posix(), path.read_text(encoding="utf-8"))]
        for name, text in items:
            inspected += 1
            if PRIVATE_PATH.search(text) or SECRET.search(text):
                problems.append((name, "private path or credential pattern"))
            if name.endswith(".ipynb"):
                book = json.loads(text)
                if any(cell.get("outputs") or cell.get("attachments") or cell.get("execution_count") is not None
                       for cell in book.get("cells", [])):
                    problems.append((name, "notebook runtime content"))
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                local = target.split("#", 1)[0].strip("<>")
                if local and not (path.parent / unquote(local)).exists():
                    problems.append((str(relative), "missing local documentation link: " + local))
    print(json.dumps({"text_objects_checked": inspected, "synthetic_binaries_checked": synthetic_binaries,
                      "problems": [{"file": name, "reason": reason} for name, reason in problems]}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
