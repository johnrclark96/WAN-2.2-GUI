"""Verify installed package versions against wan22_frozen_requirements.txt."""
from __future__ import annotations

from pathlib import Path
from typing import List

from importlib import metadata
from packaging.requirements import Requirement


REQ_FILE = Path("wan22_frozen_requirements.txt")


def parse_requirements() -> List[Requirement]:
    reqs: List[Requirement] = []
    if not REQ_FILE.exists():
        return reqs
    for line in REQ_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(Requirement(line))
    return reqs


def main() -> int:
    missing: List[str] = []
    mismatched: List[str] = []
    for req in parse_requirements():
        name = req.name
        try:
            ver = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if req.specifier and ver not in req.specifier:
            mismatched.append(f"{name} {ver} != {req.specifier}")
    if missing or mismatched:
        if missing:
            print("Missing:")
            for m in missing:
                print(f"  - {m}")
        if mismatched:
            print("Version mismatch:")
            for m in mismatched:
                print(f"  - {m}")
        return 1
    print("Environment OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
