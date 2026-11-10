import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_ROOT = _TESTS_DIR.parent
for _p in (str(_TESTS_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def clean_engines():
    from skillopt.llm import clear_engine_cache

    clear_engine_cache()
    yield
    clear_engine_cache()
