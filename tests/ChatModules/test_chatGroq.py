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

# Fixture to remove GROQ_API_KEY from environment
@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

# Fixture to set a fake GROQ_API_KEY
@pytest.fixture
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

# --- ENV VAR TESTS ---
class TestChatGroqEnv:
    def test_missing_api_key_raises_exception(self, no_api_key):
        with pytest.raises(EnvironmentError, match="Missing GROQ_API_KEY"):
            ChatGroq()
    
class TestChatGroqInvoke:
    def test_invoke_returns_expected_output(self, fake_api_key):
        # Create a mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "mocked response"
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("llm_graph.ChatModels.chatGroq.Groq") as MockGroq:
            mock_client = MockGroq.return_value
            mock_client.chat.completions.create.return_value = mock_completion

            chat = ChatGroq()
            result = chat.invoke(BaseMessage(content="Hello", role="User"))

            # Assert API call and output
            mock_client.chat.completions.create.assert_called_once_with(
                messages=[{"content": "Hello", "role": "User"}],
                model="llama-3.3-70b-versatile"
            )
            assert result == "mocked response"


