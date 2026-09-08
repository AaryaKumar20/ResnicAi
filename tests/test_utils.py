import os
import tempfile

from src.utils import (
    clean_ansi,
    load_agent_prompts,
    set_log_callback,
    clear_log_callback,
)


def test_clean_ansi():
    text = "\033[31mHello World\033[0m"

    result = clean_ansi(text)

    assert result == "Hello World"


def test_load_agent_prompts():
    prompt_content = """Agent 1:
Role: Test Researcher
Goal: Conduct research
Backstory: You are a researcher
Expected Output (Task Level): A list of researched facts

Agent 2:
Role: Test Writer
Goal: Write a report
Backstory: You are a technical writer
Expected Output (Task Level): A structured report
"""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".txt"
    ) as temp_file:
        temp_file.write(prompt_content)
        temp_path = temp_file.name

    try:
        result = load_agent_prompts(temp_path)

        assert result["agent_1"]["role"] == "Test Researcher"
        assert result["agent_1"]["goal"] == "Conduct research"
        assert result["agent_1"]["backstory"] == "You are a researcher"
        assert result["agent_1"]["expected_output"] == "A list of researched facts"

        assert result["agent_2"]["role"] == "Test Writer"
        assert result["agent_2"]["goal"] == "Write a report"

    finally:
        os.remove(temp_path)


def test_load_agent_prompts_missing_file():
    result = load_agent_prompts("file_that_does_not_exist.txt")

    assert result is None


def test_log_callback():
    callback_called = []

    def test_callback(data):
        callback_called.append(data)

    set_log_callback(test_callback)

    # Import the module itself so we can verify the callback state.
    import src.utils as utils

    assert utils.ACTIVE_CALLBACK == test_callback

    clear_log_callback()

    assert utils.ACTIVE_CALLBACK is None