# Release Workflow

Cafe releases are managed through GitHub issues, pull requests, release notes,
Git tags, and PyPI publishing. The package uses PEP 440-compatible versions. For
release candidates, follow the Scanpy-style pattern:

```text
0.2.1rc1
0.2.1rc2
0.2.1
```

Use release candidates for TestPyPI and early user validation. Promote to the
final version only after installation and smoke tests pass.

## Version Sources

Before opening a release PR, update:

- `pyproject.toml`: `project.version`
- `cafe/_settings.py`: `settings.version`
- `docs/release_notes/unreleased.md`

The CI version check runs:

```bash
python scripts/check_version.py
```

For a planned release candidate:

```bash
python scripts/check_version.py --expected 0.2.1rc1
```

## Build Check

Build from a clean tree and check both wheel and source distribution:

```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

The `Package` workflow runs the same build and `twine check` steps on pull
requests and pushes to `dev`.

## TestPyPI

Use the `Publish TestPyPI` workflow from GitHub Actions and set the expected
version, for example `0.2.1rc1`.

After the workflow succeeds, test installation in a clean environment:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --no-cache-dir \
  cafe-release==0.2.1rc1
```

Run the import check outside the repository directory so the local source tree
does not shadow the installed package.

## PyPI

Create and push a version tag only after the release PR is merged and TestPyPI
installation works:

```bash
git checkout dev
git pull github dev
git tag v0.2.1rc1
git push github v0.2.1rc1
```

Pushing a tag that starts with `v` triggers the `Publish PyPI` workflow. The
workflow strips the leading `v` and fails if the tag version does not match
`pyproject.toml`.

## PyPI Trusted Publishing

The publish workflows use GitHub Actions trusted publishing. Configure trusted
publishers in PyPI and TestPyPI for:

- repository: `HuangDDU/cafe-release`
- workflow: `.github/workflows/publish-pypi.yml`
- environment: `pypi`

For TestPyPI:

- repository: `HuangDDU/cafe-release`
- workflow: `.github/workflows/publish-testpypi.yml`
- environment: `testpypi`

Do not reuse deleted PyPI versions. PyPI permanently reserves uploaded filenames,
so a failed or deleted `0.2.1rc1` must be followed by `0.2.1rc2` or another new
version.

## PR Checklist

- [ ] Issue is linked with `Closes #...`.
- [ ] Version numbers are synchronized.
- [ ] Release notes are updated.
- [ ] `python scripts/check_version.py --expected <version>` passes.
- [ ] `python -m build` passes from a clean `dist/`.
- [ ] `python -m twine check dist/*` passes.
- [ ] TestPyPI installation is verified before creating the PyPI tag.
