from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from entireio_retrieval.config import load_config
from entireio_retrieval.openrouter import OpenRouterClient


class _SmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


@pytest.mark.live
def test_live_openrouter_structured_smoke(tmp_path: Path) -> None:
    if os.environ.get("ENTIREIO_LIVE_OPENROUTER") != "1":
        pytest.skip("set ENTIREIO_LIVE_OPENROUTER=1 to allow the opt-in API call")
    config_path = Path(__file__).parents[1] / "config" / "default.yaml"
    config = load_config(config_path)
    result = OpenRouterClient(
        config.openrouter,
        config.key_file,
        tmp_path / "cache",
    ).complete_structured(
        messages=[
            {
                "role": "system",
                "content": "Return status=ok. Do not include any other content.",
            },
            {"role": "user", "content": "Structured API connectivity smoke test."},
        ],
        response_model=_SmokeResponse,
        purpose="live-connectivity-smoke",
    )
    assert result.status == "ok"
