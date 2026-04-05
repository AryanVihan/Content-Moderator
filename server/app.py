"""app.py — OpenEnv entry point for MetaModEnv.

This module satisfies the OpenEnv server entry point convention:
  [project.scripts] server = "server.app:main"
"""
import uvicorn
from server.main import app  # noqa: F401


def main() -> None:
    """Start the MetaModEnv FastAPI server."""
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
