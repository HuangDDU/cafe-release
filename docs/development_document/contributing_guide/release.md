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

The `Release` workflow is triggered only by a manual run or by pushing a
three-part version tag. Cafe tags use the PEP 440 version directly, such as
`0.2.1rc1`, without a leading `v`. The workflow runs the release pipeline in
this order:

1. Build wheel and source distribution.
2. Run `twine check`.
3. Validate the release version against PEP 440 and Cafe's package metadata.
4. Publish to TestPyPI.
5. Publish to PyPI.

## TestPyPI

Use the `Release` workflow from GitHub Actions and set the expected version, for
example `0.2.1rc1`. The workflow builds and checks the package, validates that
the requested version matches `pyproject.toml`, then publishes to TestPyPI
before continuing to PyPI.

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

Create and push a version tag only after the release PR is merged:

```bash
git checkout dev
git pull github dev
git tag 0.2.1rc1
git push github 0.2.1rc1
```

Pushing a version tag such as `0.2.1rc1` triggers the full `Release` workflow.
The workflow validates the tag as PEP 440 and fails if the tag version does not
match `pyproject.toml`.

If you want a manual approval gate between TestPyPI and PyPI, configure GitHub's
`pypi` environment with required reviewers. The workflow will pause before the
PyPI job starts.

## PyPI Trusted Publishing

The publish workflows use GitHub Actions trusted publishing. Configure trusted
publishers in PyPI and TestPyPI for:

- repository: `HuangDDU/cafe-release`
- workflow: `.github/workflows/release.yml`
- environment: `pypi`

For TestPyPI:

- repository: `HuangDDU/cafe-release`
- workflow: `.github/workflows/release.yml`
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
- [ ] GitHub `pypi` environment approval is configured if manual review is required before PyPI publishing.
