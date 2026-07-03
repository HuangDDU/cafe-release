from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def runtime_version() -> str:
    tree = ast.parse((ROOT / "cafe" / "_settings.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "version":
                    return str(node.value.value)
    raise RuntimeError("Could not find cafe runtime version.")


def check_version(expected: str | None = None) -> list[str]:
    project = pyproject_version()
    runtime = runtime_version()
    errors: list[str] = []

    for label, version in {"pyproject": project, "runtime": runtime, "expected": expected}.items():
        if version:
            try:
                Version(version)
            except InvalidVersion:
                errors.append(f"{label} version {version!r} is not PEP 440 compatible.")

    if runtime != project:
        errors.append(f"runtime version {runtime!r} != pyproject version {project!r}.")
    if expected and expected != project:
        errors.append(f"expected version {expected!r} != pyproject version {project!r}.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected")
    args = parser.parse_args()

    errors = check_version(args.expected)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(f"Cafe package version is consistent: {pyproject_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
