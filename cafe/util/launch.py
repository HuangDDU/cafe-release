from __future__ import annotations

import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Sequence

from cafe._logging import logger


def build_cellxgene_launch_command(
    adata_filename: str | Path,
    *,
    port: int = 5005,
    use_cellxgene_cafe: bool = False,
    conda_env: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build the command used to launch native cellxgene or cafe-cellxgene."""

    launcher = "cafe-cellxgene" if use_cellxgene_cafe else "cellxgene"
    command = [launcher, "launch", str(adata_filename), "--port", str(port)]
    if extra_args:
        command.extend(str(item) for item in extra_args)
    if conda_env:
        return ["conda", "run", "-n", str(conda_env), "--no-capture-output", *command]
    return command


def launch_cellxgene(
    adata_filename: str | Path,
    *,
    port: int = 5005,
    use_cellxgene_cafe: bool = False,
    conda_env: str | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    startup_wait: float = 5.0,
    extra_args: Sequence[str] | None = None,
):
    """Launch cellxgene or cafe-cellxgene for an existing h5ad file.

    Parameters
    ----------
    adata_filename
        Path to a readable h5ad file.
    port
        HTTP port passed to the launcher.
    use_cellxgene_cafe
        If true, run ``cafe-cellxgene launch``. Otherwise run native
        ``cellxgene launch``.
    conda_env
        Optional conda environment. When provided, the command is wrapped with
        ``conda run -n <env> --no-capture-output``.
    open_browser
        Open the local browser after the process appears to be running.
    host
        Browser host used to build the URL.
    startup_wait
        Seconds to wait before opening the browser.
    extra_args
        Extra arguments appended to the underlying launcher command.
    """

    adata_path = Path(adata_filename)
    if not adata_path.exists():
        raise FileNotFoundError(f"h5ad file does not exist: {adata_path}")

    command = build_cellxgene_launch_command(
        adata_path,
        port=port,
        use_cellxgene_cafe=use_cellxgene_cafe,
        conda_env=conda_env,
        extra_args=extra_args,
    )
    launcher_name = "cafe-cellxgene" if use_cellxgene_cafe else "cellxgene"
    logger.info(f"Launch {launcher_name} with command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _stream_pipe(process.stdout, "[cellxgene stdout] ")
    _stream_pipe(process.stderr, "[cellxgene stderr] ")

    if open_browser:
        time.sleep(startup_wait)
        if process.poll() is None:
            url = f"http://{host}:{port}"
            logger.info("Cellxgene server started at: %s", url)
            webbrowser.open(url)

    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping cellxgene server")
        process.terminate()
        process.wait()
    return process


def _stream_pipe(pipe, prefix: str) -> None:
    if pipe is None:
        return

    def print_output() -> None:
        for line in iter(pipe.readline, ""):
            if line:
                logger.debug(f"{prefix}{line.rstrip()}")
        pipe.close()

    threading.Thread(target=print_output, daemon=True).start()
