"""Temporarily exclude method function integration tests from collection.

These tests depend on method-specific environments and will be re-enabled when
the method test matrix is stabilized.
"""

collect_ignore_glob = ["test_*.py"]
