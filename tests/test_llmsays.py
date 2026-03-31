import os
import pytest
from unittest.mock import MagicMock, patch
from llmsays import cli, llmsays

@patch.dict(os.environ, {"GROQ_API_KEY": "test_key"})
@patch("llmsays.get_tier", return_value="small")
@patch("llmsays._get_client")
def test_llmsays(mock_client, _mock_tier):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Mock output"
    mock_client.return_value.chat.completions.create.return_value = mock_resp
    assert llmsays("Test") == "Mock output"


@patch.dict(os.environ, {}, clear=True)
@patch("llmsays.get_tier", return_value="small")
def test_llmsays_raises_when_no_provider_keys(_mock_tier):
    with pytest.raises(RuntimeError):
        llmsays("Test")

def test_cli(capsys):
    with patch("llmsays.llmsays", return_value="CLI Mock"):
        with patch("sys.argv", ["llmsays", "Test query"]):
            cli()
    captured = capsys.readouterr()
    assert "CLI Mock" in captured.out