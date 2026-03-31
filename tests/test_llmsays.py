import os
import pytest
from unittest.mock import MagicMock, patch
import llmsays as module
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


def test_latency_sorted_providers_prefers_lower_latency():
    module._LATENCY_MS["small"] = {"Openrouter": 110.0, "Groq": 40.0}
    ordered = module._latency_sorted_providers("small", ["Openrouter", "Groq", "NIM"])
    assert ordered == ["Groq", "Openrouter", "NIM"]


@patch("llmsays.get_tier", return_value="small")
@patch("llmsays._provider_order", return_value=["Groq", "Openrouter"])
@patch("llmsays._parallel_llmsays", return_value="parallel response")
def test_llmsays_parallel_mode(mock_parallel, _mock_order, _mock_tier):
    text = llmsays("hello", use_multiprocessing=True)
    assert text == "parallel response"
    mock_parallel.assert_called_once()


@patch("llmsays.get_tier", return_value="small")
@patch("llmsays._provider_order", return_value=["Groq", "Openrouter"])
@patch("llmsays._call_provider")
def test_llmsays_sequential_failover(mock_call_provider, _mock_order, _mock_tier):
    mock_call_provider.side_effect = [RuntimeError("groq down"), "ok from openrouter"]
    text = llmsays("hello")
    assert text == "ok from openrouter"

def test_cli(capsys):
    with patch("llmsays.llmsays", return_value="CLI Mock"):
        with patch("sys.argv", ["llmsays", "Test query"]):
            cli()
    captured = capsys.readouterr()
    assert "CLI Mock" in captured.out