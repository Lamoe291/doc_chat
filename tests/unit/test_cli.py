import pytest
from unittest.mock import patch
from pathlib import Path
import sys
from doc_chat.cli import main


def test_missing_command():
    with pytest.raises(SystemExit):
        main([])

def test_unknown_command():
    with pytest.raises(SystemExit):
        main(["something"])

def test_chat_requires_index():
    with pytest.raises(SystemExit):
        main(["chat"])

def test_index_command_is_called():
    with patch("doc_chat.cli.index_command") as mock_index:
        main(["index", "document.pdf"])

        mock_index.assert_called_once()
        args = mock_index.call_args.args[0]

        assert args.pdf_path == Path("document.pdf")

def test_chat_command_is_called():
    with patch("doc_chat.cli.chat_command") as mock_chat:
        main(["chat", "--index", "my_document"])

        mock_chat.assert_called_once()
        args = mock_chat.call_args.args[0]

        assert args.index == "my_document"
