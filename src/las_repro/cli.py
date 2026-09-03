"""Process entry points for the local LAS-compatible service."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import ValidationError

from .api import create_app
from .config import Settings
from .store import SQLiteTaskStore


Command = Callable[[argparse.Namespace, Settings, threading.Event], int]


class _SignalShutdown(KeyboardInterrupt):
    """Cooperative process signal that must unwind the active lease boundary."""


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one explicit process role and run it to a clean lifecycle boundary."""
    stop = threading.Event()
    try:
        with _stop_on_signals(stop):
            parser = _parser()
            arguments = parser.parse_args(argv)
            _notify_signal_ready()
            try:
                settings = Settings.from_env()
            except ValidationError:
                print("las-repro: invalid LAS_ configuration", file=sys.stderr)
                return 2
            command: Command = arguments.command
            return command(arguments, settings, stop)
    except _SignalShutdown:
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"las-repro: {type(error).__name__}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="las-repro",
        description="Run one local LAS-compatible service process role.",
    )
    commands = parser.add_subparsers(dest="role", required=True)

    init_db = commands.add_parser("init-db", help="initialize the SQLite task store")
    init_db.set_defaults(command=_init_db)

    api = commands.add_parser("api", help="serve only the Submit/Poll control plane")
    _listen_options(api)
    api.set_defaults(command=_api)

    coordinator = commands.add_parser(
        "coordinator",
        help="resolve media and coordinate pipelines without loading a GPU model",
    )
    coordinator.add_argument("--worker-id", default="coordinator")
    coordinator.add_argument("--once", action="store_true", help="claim at most one task")
    coordinator.set_defaults(command=_coordinator)

    gpu_worker = commands.add_parser(
        "gpu-worker",
        help="load one local model on one explicit CUDA device",
    )
    gpu_worker.add_argument("--device", type=int, required=True, choices=range(0, 1024))
    gpu_worker.add_argument("--worker-id")
    gpu_worker.add_argument("--model-name", default="qwen3-vl-8b-instruct")
    gpu_worker.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    gpu_worker.add_argument("--once", action="store_true", help="claim at most one job")
    gpu_worker.set_defaults(command=_gpu_worker)

    run_fake = commands.add_parser(
        "run-fake",
        help="run an API-compatible local stack with no model weights or GPU",
    )
    _listen_options(run_fake)
    run_fake.add_argument(
        "--once",
        action="store_true",
        help="drain currently claimable local tasks and exit without serving HTTP",
    )
    run_fake.set_defaults(command=_run_fake)
    return parser


def _listen_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help="override LAS_API_HOST")
    parser.add_argument("--port", type=int, choices=range(1, 65536), help="override LAS_API_PORT")


@contextmanager
def _store(settings: Settings) -> Iterator[SQLiteTaskStore]:
    store = SQLiteTaskStore(settings.database_path)
    try:
        store.initialize()
        yield store
    finally:
        store.close()


def _init_db(
    _: argparse.Namespace,
    settings: Settings,
    __: threading.Event,
) -> int:
    with _store(settings):
        return 0


def _api(
    arguments: argparse.Namespace,
    settings: Settings,
    stop: threading.Event,
) -> int:
    with _store(settings) as store:
        app = create_app(settings, store)
        _serve(app, _host(arguments, settings), _port(arguments, settings), stop=stop)
    return 0


def _coordinator(
    arguments: argparse.Namespace,
    settings: Settings,
    stop: threading.Event,
) -> int:
    # Keep worker and pipeline imports inside this role.  In particular, this
    # path never imports the optional Qwen/PyTorch/Transformers backend.
    from .media import MediaResolver, TosAdapter
    from .workers import Coordinator

    with _store(settings) as store:
        coordinator = Coordinator(
            store,
            MediaResolver(settings, tos_adapter=TosAdapter(settings)),
            settings,
            _pipeline_registry(),
            worker_id=arguments.worker_id,
        )
        if arguments.once:
            coordinator.run_once()
        else:
            coordinator.run_forever(stop)
    return 0


def _gpu_worker(
    arguments: argparse.Namespace,
    settings: Settings,
    stop: threading.Event,
) -> int:
    if settings.backend != "qwen3_vl":
        raise ValueError("gpu-worker requires LAS_BACKEND=qwen3_vl")
    if arguments.device not in settings.gpu_devices:
        raise ValueError("gpu-worker device is absent from LAS_GPU_DEVICES")
    # This is the only production CLI role that imports or loads the optional
    # GPU backend.  One invocation constructs exactly one model for one device.
    from .models.qwen3_vl import Qwen3VLModel
    from .workers import GPUWorker

    device = f"cuda:{arguments.device}"
    worker_id = arguments.worker_id or f"gpu-{arguments.device}"
    with _store(settings) as store:
        model = Qwen3VLModel.load_alias(
            arguments.model_name,
            settings.model_registry,
            device,
            arguments.dtype,
            max_output_chars=settings.max_model_output_chars,
        )
        worker = GPUWorker(
            store,
            model,
            worker_id,
            device,
            model_name=arguments.model_name,
            lease_seconds=settings.lease_seconds,
        )
        try:
            if arguments.once:
                worker.run_once()
            else:
                worker.run_forever(stop)
        finally:
            worker.close()
    return 0


def _run_fake(
    arguments: argparse.Namespace,
    settings: Settings,
    stop: threading.Event,
) -> int:
    from .media import MediaResolver, TosAdapter
    from .models.fake import FakeVideoModel
    from .workers import Coordinator, GPUWorker

    with _store(settings) as store:
        resolver = MediaResolver(settings, tos_adapter=TosAdapter(settings))
        coordinator = Coordinator(
            store,
            resolver,
            settings,
            _pipeline_registry(),
            worker_id="fake-coordinator",
        )
        workers = [
            GPUWorker(
                store,
                FakeVideoModel(),
                worker_id=f"fake-gpu-{model_name}",
                device=f"fake:{index}",
                model_name=model_name,
                lease_seconds=settings.lease_seconds,
            )
            for index, model_name in enumerate(sorted(settings.model_registry))
        ]
        gpu_stops = [threading.Event() for _ in workers]
        gpu_threads = [
            _worker_thread(
                f"las-fake-gpu-{index}",
                worker.run_forever,
                gpu_stop,
            )
            for index, (worker, gpu_stop) in enumerate(
                zip(workers, gpu_stops, strict=True)
            )
        ]
        try:
            for gpu_thread in gpu_threads:
                gpu_thread.start()
            if arguments.once:
                while coordinator.run_once():
                    pass
            else:
                coordinator_thread = _worker_thread(
                    "las-fake-coordinator",
                    coordinator.run_forever,
                    stop,
                )
                try:
                    coordinator_thread.start()
                    app = create_app(settings, store)
                    _serve(
                        app,
                        _host(arguments, settings),
                        _port(arguments, settings),
                        stop=stop,
                    )
                finally:
                    stop.set()
                    if coordinator_thread.ident is not None:
                        coordinator_thread.join()
        finally:
            # Keep inference available until the active coordinator claim has
            # finished.  It may still be waiting for jobs created before the
            # API/coordinator stop gate closed.
            try:
                for gpu_stop in gpu_stops:
                    gpu_stop.set()
                for gpu_thread in gpu_threads:
                    if gpu_thread.ident is not None:
                        gpu_thread.join()
            finally:
                for worker in workers:
                    worker.close()
    return 0


def _pipeline_registry() -> Any:
    from .pipelines.base import PipelineRegistry
    from .pipelines.embodied import EmbodiedActionPipeline, EmbodiedActiveObjectsPipeline
    from .pipelines.general import GeneralCaptionPipeline

    registry = PipelineRegistry()
    registry.register("general_video_captioning", GeneralCaptionPipeline)
    registry.register("embodied_active_object_detection", EmbodiedActiveObjectsPipeline)
    registry.register("embodied_action_captioning", EmbodiedActionPipeline)
    return registry


def _worker_thread(
    name: str,
    target: Callable[[threading.Event], None],
    stop: threading.Event,
) -> threading.Thread:
    return threading.Thread(target=target, args=(stop,), name=name, daemon=False)


def _serve(
    app: Any,
    host: str,
    port: int,
    *,
    stop: threading.Event | None = None,
) -> None:
    import uvicorn

    class CoordinatedServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            del sig, frame
            if stop is not None:
                stop.set()
            # Uvicorn records handled signals and re-raises them after its
            # graceful shutdown.  This CLI owns the process lifecycle, so
            # setting the exit flag directly avoids turning a completed
            # shutdown into a negative signal exit status.
            self.should_exit = True

    server = CoordinatedServer(uvicorn.Config(app, host=host, port=port))
    server.run()


@contextmanager
def _stop_on_signals(stop: threading.Event) -> Iterator[None]:
    """Turn TERM/INT into a cooperative stop before the next lease claim."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handled = tuple(
        item
        for item in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None))
        if item is not None
    )
    mask_signals = getattr(signal, "pthread_sigmask", None)
    entry_mask: set[signal.Signals] | None = None
    if callable(mask_signals):
        entry_mask = mask_signals(signal.SIG_BLOCK, handled)
    previous = {item: signal.getsignal(item) for item in handled}

    def request_stop(_: int, __: Any) -> None:
        if stop.is_set():
            return
        stop.set()
        # Unwind synchronous worker code so its BaseException boundary expires
        # the exact owner/generation before this process exits.
        raise _SignalShutdown

    try:
        for item in handled:
            signal.signal(item, request_stop)
        if entry_mask is not None:
            mask_signals(signal.SIG_SETMASK, entry_mask)
        yield
    finally:
        try:
            if callable(mask_signals):
                mask_signals(signal.SIG_BLOCK, handled)
        finally:
            try:
                for item, handler in previous.items():
                    signal.signal(item, handler)
            finally:
                if callable(mask_signals) and entry_mask is not None:
                    mask_signals(signal.SIG_SETMASK, entry_mask)


def _notify_signal_ready() -> None:
    """Notify a supervisor that cooperative TERM/INT handling is installed."""
    raw_descriptor = os.environ.get("_LAS_REPRO_SIGNAL_READY_FD")
    if raw_descriptor is None:
        return
    descriptor = int(raw_descriptor)
    try:
        os.write(descriptor, b"R")
    finally:
        os.close(descriptor)


def _host(arguments: argparse.Namespace, settings: Settings) -> str:
    return arguments.host or settings.api_host


def _port(arguments: argparse.Namespace, settings: Settings) -> int:
    return arguments.port or settings.api_port


if __name__ == "__main__":
    raise SystemExit(main())
