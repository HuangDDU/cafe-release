import sys

import cafe.cli


def test_cli_launch_cellxgene_delegates_to_launcher(monkeypatch):
    calls = []

    def fake_launch_cellxgene(filename, **kwargs):
        calls.append((filename, kwargs))
        return None

    monkeypatch.setattr("cafe.util.launch.launch_cellxgene", fake_launch_cellxgene)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cafe",
            "launch-cellxgene",
            "dataset.h5ad",
            "--port",
            "5012",
            "--cellxgene-cafe",
            "--conda-env",
            "cafe",
            "--no-open",
        ],
    )

    cafe.cli.main()

    assert calls == [
        (
            "dataset.h5ad",
            {
                "port": 5012,
                "use_cellxgene_cafe": True,
                "conda_env": "cafe",
                "open_browser": False,
            },
        )
    ]
