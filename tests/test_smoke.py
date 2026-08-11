"""Smoke tests for the caracal_rag package skeleton."""

import caracal_rag


def test_package_imports():
    assert caracal_rag is not None


def test_version_is_string():
    assert isinstance(caracal_rag.__version__, str)


def test_version_value():
    assert caracal_rag.__version__ == "0.1.0"
