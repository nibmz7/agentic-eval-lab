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


@pytest.fixture(autouse=True)
def tracing_off():
    """No test sends traces anywhere, even with Langfuse keys in the environment."""
    from support_lab import tracing

    tracing.configure(None)
    yield
    tracing.configure(None)


@pytest.fixture(scope="session")
def _span_exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from support_lab import tracing

    exporter = InMemorySpanExporter()
    # Spans go to the exporter, never over the network. The URL is unreachable on purpose.
    tracing.new_client(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
    )
    return exporter


@pytest.fixture
def traced(_span_exporter):
    """Turn tracing on for one test. Call the result to get the finished spans."""
    from support_lab import tracing

    _span_exporter.clear()
    tracing.configure("pk-lf-test")

    def finished():
        tracing.flush()
        return list(_span_exporter.get_finished_spans())

    return finished
