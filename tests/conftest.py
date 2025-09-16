# tests/conftest.py
import os, sys, pathlib
import pytest

# Add ./src to sys.path so `import mancala_ai...` works in tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Optionally point model registry to local default
os.environ.setdefault("MODEL_REGISTRY", str(ROOT / "model_registry" / "latest"))

@pytest.fixture(scope="session")
def app():
    from mancala_ai.api.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return app

@pytest.fixture(scope="session")
def client(app):
    return app.test_client()
