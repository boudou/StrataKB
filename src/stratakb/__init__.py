"""StrataKB: a layered, self-learning knowledge base served over MCP."""

from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml is the one place the version is written
    __version__ = version("stratakb")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"
