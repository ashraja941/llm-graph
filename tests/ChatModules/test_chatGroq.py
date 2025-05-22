import os
import shutil
from llm_graph.messages.base import BaseMessage
import pytest
from unittest.mock import MagicMock, patch
from llm_graph.ChatModels.chatGroq import ChatGroq 

# Fixture to handle .env backup/removal and restoration
@pytest.fixture(autouse=True)
def backup_and_restore_env_file():
    env_path = ".env"
    env_backup_path = ".env.bak"

    if os.path.exists(env_path):
        shutil.copy(env_path, env_backup_path)
        os.remove(env_path)

    yield

    if os.path.exists(env_backup_path):
        shutil.move(env_backup_path, env_path)

# Fixture to remove OPENAI_API_KEY from environment
@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

# Fixture to set a fake OPENAI_API_KEY
@pytest.fixture
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

# --- ENV VAR TESTS ---
class TestChatOpenAIEnv:
    def test_missing_api_key_raises_exception(self, no_api_key):
        with pytest.raises(EnvironmentError, match="Missing GROQ_API_KEY"):
            ChatGroq()
    
class TestChatOpenAIInvoke:
    def test_invoke_returns_expected_output(self, fake_api_key):
        # Create a mock response with .output_text
        mock_response = MagicMock()
        mock_response.output_text = "mocked response"

        with patch("llm_graph.ChatModels.chatGroq.ChatGroq") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.responses.create.return_value = mock_response

            chat = ChatGroq()
            result = chat.invoke(BaseMessage(content="Hi",role="User"))

            # Assert API call and output
            mock_client.responses.create.assert_called_once_with(
                model="llama-3.3-70b-versatile",
                input="Hello"
            )
            assert result == "mocked response"
