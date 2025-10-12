import os
import pytest
from unittest.mock import MagicMock, patch

from llmsays import cli, llmsays


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
@patch("llmsays._get_client")
def test_llmsays(mock_client):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Mock output"
    mock_client.return_value.chat.completions.create.return_value = (
        mock_resp
    )

    assert llmsays("Test") == "Mock output"


def test_cli(capsys):
    with patch("llmsays.llmsays", return_value="CLI Mock"):
        with patch(
            "sys.argv", ["llmsays", "Test query"]
        ):
            cli()
    captured = capsys.readouterr()
    assert "CLI Mock" in captured.out