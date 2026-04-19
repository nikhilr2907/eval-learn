"""Pytest configuration for eval-learn tests."""



def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
