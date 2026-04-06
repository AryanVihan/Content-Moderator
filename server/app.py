"""app.py — OpenEnv entry point for MetaModEnv.

This module satisfies the OpenEnv server entry point convention:
  entry_point: "server.app:main"
"""
import os

import uvicorn


def main() -> None:
    """
    OpenEnv entry point. Called by openenv validate and openenv run.
    Starts the FastAPI server on port 7860.
    """
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,
        timeout_keep_alive=30,
        access_log=True,
    )


if __name__ == "__main__":
    main()
