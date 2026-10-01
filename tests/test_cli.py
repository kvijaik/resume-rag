"""CLI: single-query and interactive modes."""

import sys

from app import cli


def test_single_query_mode(vectorstore, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli", "Need", "a", "React", "frontend", "developer"])
    cli.main()
    out = capsys.readouterr().out
    assert "Status: matched" in out
    assert "react_ui_developer.txt" in out


def test_interactive_mode_until_quit(vectorstore, monkeypatch, capsys):
    replies = iter(["help", "QUIT"])
    monkeypatch.setattr(sys, "argv", ["cli"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    cli.main()
    out = capsys.readouterr().out
    assert "[unclear]" in out
