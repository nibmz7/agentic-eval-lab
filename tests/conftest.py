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


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def run_app():
    """Start ASGI apps on real local ports, in background threads, for the test."""
    import threading
    import time

    import uvicorn

    servers = []

    def start(app, port: int) -> None:
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        servers.append((server, thread))

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)
