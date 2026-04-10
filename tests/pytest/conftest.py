import pytest  # noqa: F401

pytest_plugins = [
    "fixtures.config",
    "fixtures.data",
    "fixtures.datasets",
    "fixtures.mocks",
    "fixtures.storage",
]
