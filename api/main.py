from __future__ import annotations

import json


def _response(status: int, payload: dict[str, str]) -> tuple[int, list[tuple[str, str]], bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    return status, headers, body


def app(environ, start_response):
    """Compatibility WSGI entrypoint for Vercel Python detection.

    Main production endpoint remains /api/ask (implemented in api/ask.js).
    """
    path = environ.get("PATH_INFO", "")

    if path in {"/api/main", "/api/main/", "/api/health", "/api/health/"}:
        status_code, headers, body = _response(
            200,
            {
                "status": "ok",
                "message": "Python entrypoint detected. Use POST /api/ask for Climate AI responses.",
            },
        )
    else:
        status_code, headers, body = _response(
            404,
            {
                "error": "Not found",
                "hint": "Use POST /api/ask for AI queries.",
            },
        )

    reason = "OK" if status_code == 200 else "Not Found"
    start_response(f"{status_code} {reason}", headers)
    return [body]