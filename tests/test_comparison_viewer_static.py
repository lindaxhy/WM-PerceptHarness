from __future__ import annotations

import json
import threading
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class _ShellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.assets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        for name in ("href", "src"):
            if attributes.get(name):
                self.assets.append(attributes[name] or "")


def _has_tag(
    parser: _ShellParser, tag: str, **attributes: str
) -> bool:
    return any(
        candidate == tag
        and all(values.get(name) == value for name, value in attributes.items())
        for candidate, values in parser.tags
    )


def test_viewer_route_and_all_committed_assets_resolve_over_http() -> None:
    repository = Path(__file__).resolve().parents[1]
    handler = partial(_QuietHandler, directory=str(repository))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/"
    try:
        with urlopen(f"{base_url}evaluation/viewer/", timeout=5) as response:
            assert response.status == 200
            html = response.read().decode("utf-8")

        parser = _ShellParser()
        parser.feed(html)
        assert _has_tag(parser, "main", id="comparison-workspace")
        assert _has_tag(parser, "nav", **{"aria-label": "Demo samples"})
        assert _has_tag(parser, "video", id="demo-video")
        assert _has_tag(parser, "input", id="video-file", type="file")
        assert _has_tag(parser, "section", id="las-panel")
        assert _has_tag(parser, "section", id="local-panel")
        assert _has_tag(parser, "section", id="timeline-panel")
        assert _has_tag(parser, "div", id="status", role="alert")
        assert not any(asset.startswith(("http://", "https://", "//")) for asset in parser.assets)

        expected_assets = [
            "evaluation/viewer/styles.css",
            "evaluation/viewer/js/app.js",
            "evaluation/viewer/js/model.js",
            "evaluation/viewer/data/demo-manifest.json",
        ]
        manifest = json.loads(
            urlopen(f"{base_url}evaluation/viewer/data/demo-manifest.json", timeout=5)
            .read()
            .decode("utf-8")
        )
        for sample in manifest["samples"]:
            expected_assets.extend([sample["las_path"], sample["local_path"]])
        for path in expected_assets:
            with urlopen(f"{base_url}{path}", timeout=5) as response:
                assert response.status == 200, path
                assert response.read(), path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
