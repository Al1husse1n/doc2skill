import pytest


@pytest.fixture(scope="module")
def vcr_config():
    """Configuration for pytest-recording / vcrpy cassettes."""
    return {
        "filter_headers": ["authorization", "x-goog-api-key"],
        "ignore_localhost": True,
    }

