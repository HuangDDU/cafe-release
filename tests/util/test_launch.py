from pathlib import Path

import pytest

from cafe.util.launch import build_cellxgene_launch_command, launch_cellxgene


def test_build_native_cellxgene_command() -> None:
    command = build_cellxgene_launch_command("dataset.h5ad", port=5005)

    assert command == ["cellxgene", "launch", "dataset.h5ad", "--port", "5005"]


def test_build_cafe_cellxgene_command() -> None:
    command = build_cellxgene_launch_command("dataset.h5ad", port=5006, use_cellxgene_cafe=True)

    assert command == ["cafe-cellxgene", "launch", "dataset.h5ad", "--port", "5006"]


def test_build_command_can_run_through_conda_env() -> None:
    command = build_cellxgene_launch_command("dataset.h5ad", port=5007, use_cellxgene_cafe=True, conda_env="cafe")

    assert command == [
        "conda",
        "run",
        "-n",
        "cafe",
        "--no-capture-output",
        "cafe-cellxgene",
        "launch",
        "dataset.h5ad",
        "--port",
        "5007",
    ]


def test_launch_cellxgene_starts_process_without_shell(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.h5ad"
    dataset.write_text("fake", encoding="utf-8")
    popen_calls = []

    class FakeProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    process = launch_cellxgene(dataset, port=5010, use_cellxgene_cafe=True, open_browser=False)

    assert isinstance(process, FakeProcess)
    assert popen_calls[0][0] == ["cafe-cellxgene", "launch", str(dataset), "--port", "5010"]
    assert popen_calls[0][1]["shell"] is False


def test_launch_cellxgene_rejects_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        launch_cellxgene(tmp_path / "missing.h5ad", open_browser=False)
