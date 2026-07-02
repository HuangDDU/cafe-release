from __future__ import annotations

import argparse
import ast
import importlib.metadata
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def read_pyproject_version(pyproject_path: Path = ROOT / "pyproject.toml") -> str:
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def read_runtime_version() -> str:
    settings_path = ROOT / "cafe" / "_settings.py"
    tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr == "version"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise RuntimeError("Could not find self.version in cafe/_settings.py.")


def read_installed_metadata_version(package_name: str = "cafe-release") -> str:
    return importlib.metadata.version(package_name)


def validate_versions(expected: str | None = None, check_metadata: bool = False) -> list[str]:
    pyproject_version = read_pyproject_version()
    runtime_version = read_runtime_version()
    errors: list[str] = []

    if expected is not None and pyproject_version != expected:
        errors.append(f"pyproject.toml version is {pyproject_version!r}, expected {expected!r}.")

    if runtime_version != pyproject_version:
        errors.append(f"cafe.settings.version is {runtime_version!r}, expected {pyproject_version!r}.")

    if check_metadata:
        metadata_version = read_installed_metadata_version()
        if metadata_version != pyproject_version:
            errors.append(f"installed metadata version is {metadata_version!r}, expected {pyproject_version!r}.")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Cafe package version metadata.")
    parser.add_argument("--expected", help="Expected release version, for example 0.2.1rc1.")
    parser.add_argument(
        "--check-installed-metadata",
        action="store_true",
        help="Also compare the installed package metadata version.",
    )
    args = parser.parse_args(argv)

    errors = validate_versions(expected=args.expected, check_metadata=args.check_installed_metadata)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Cafe package version is consistent: {read_pyproject_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
