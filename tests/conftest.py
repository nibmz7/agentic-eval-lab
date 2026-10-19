import pytest

from support_lab.backend import SupportStore


@pytest.fixture
def store():
    s = SupportStore.seeded()
    yield s
    s.close()


@pytest.fixture
def anyio_backend():
    return "asyncio"
