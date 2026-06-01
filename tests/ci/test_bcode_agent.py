from __future__ import annotations

import base64
import json

import pytest

from browser_use.bcode import Agent, BcodeNotInstalledError


class _FakeChat:
	def __init__(self, model: str | None = None, api_key: str | None = None):
		self.model = model
		self.api_key = api_key


def _llm_class(name: str) -> type[_FakeChat]:
	return type(name, (_FakeChat,), {})


class _Browser:
	cdp_url = 'ws://127.0.0.1:9222/devtools/browser/test'


def test_model_id_includes_bcode_provider_prefix():
	llm = _llm_class('ChatAnthropic')(model='claude-sonnet-4-5')
	assert Agent(task='x', llm=llm, browser=_Browser())._model_id() == 'anthropic/claude-sonnet-4-5'


def test_cdp_and_provider_key_are_forwarded_to_bcode_env(tmp_path):
	llm = _llm_class('ChatOpenAI')(model='gpt-5.5', api_key='sk-test')
	env = Agent(task='x', llm=llm, browser=_Browser())._env_overrides(_Browser.cdp_url, tmp_path)
	assert env['BU_CDP_WS'] == _Browser.cdp_url
	assert env['BU_CDP_URL'] == _Browser.cdp_url
	assert env['OPENAI_API_KEY'] == 'sk-test'
	assert env['OPENCODE_CONFIG_DIR'] == str(tmp_path / '.bcode')


def test_missing_bcode_binary_raises(monkeypatch):
	monkeypatch.delenv('BROWSER_USE_BCODE_BINARY', raising=False)
	monkeypatch.delenv('BROWSER_USE_BCODE_COMMAND', raising=False)
	monkeypatch.delenv('BCODE_BINARY', raising=False)
	monkeypatch.delenv('BCODE_BIN_PATH', raising=False)
	monkeypatch.setenv('PATH', '')
	with pytest.raises(BcodeNotInstalledError):
		from browser_use.bcode.service import find_bcode_binary

		find_bcode_binary()


def test_command_override_supports_source_checkout(monkeypatch):
	monkeypatch.setenv('BROWSER_USE_BCODE_COMMAND', 'bun run --cwd /repo/packages/opencode --conditions=browser ./src/index.ts')
	from browser_use.bcode.service import _bcode_command

	assert _bcode_command() == ['bun', 'run', '--cwd', '/repo/packages/opencode', '--conditions=browser', './src/index.ts']


@pytest.mark.asyncio
async def test_bcode_agent_parses_json_stream_and_returns_history(tmp_path, monkeypatch):
	fake = tmp_path / 'bcode'
	capture = tmp_path / 'capture.json'
	fake.write_text(
		"""#!/usr/bin/env python3
import json, os, sys
capture = os.environ["BCODE_FAKE_CAPTURE"]
if sys.argv[1] == "run":
    with open(capture, "w") as f:
        json.dump({"argv": sys.argv, "BU_CDP_WS": os.environ.get("BU_CDP_WS"), "cwd": os.getcwd()}, f)
if sys.argv[1] == "export":
    raise SystemExit(0)
session = "sess-bcode"
events = [
    {"type": "step_start", "timestamp": 1, "sessionID": session, "part": {"id": "s1"}},
    {"type": "tool_use", "timestamp": 2, "sessionID": session, "part": {"id": "t1", "type": "tool", "tool": "browser_execute", "state": {"status": "completed", "input": {"code": "await session.connect()"}, "output": "title: Example"}}},
    {"type": "text", "timestamp": 3, "sessionID": session, "part": {"id": "m1", "type": "text", "text": "Example Domain summary"}},
    {"type": "step_finish", "timestamp": 4, "sessionID": session, "part": {"id": "s2"}},
]
for event in events:
    print(json.dumps(event), flush=True)
""",
	)
	fake.chmod(0o755)
	monkeypatch.setenv('BROWSER_USE_BCODE_BINARY', str(fake))
	monkeypatch.setenv('BCODE_FAKE_CAPTURE', str(capture))

	result = await Agent(
		task='Go to example.com and summarize it',
		llm=_llm_class('ChatOpenAI')(model='gpt-5.5'),
		browser=_Browser(),
		workspace_dir=tmp_path / 'workspace',
	).run(max_steps=50)

	assert result.exit_code == 0
	assert result.session_id == 'sess-bcode'
	assert result.final_result() == 'Example Domain summary'
	assert result.is_done()
	assert result.is_successful() is True
	assert result.action_results()[0]['output'] == 'title: Example'
	assert result.history[0].result[0].extracted_content == 'title: Example'
	captured = json.loads(capture.read_text())
	assert captured['BU_CDP_WS'] == _Browser.cdp_url
	assert captured['argv'][1:5] == ['run', '--format', 'json', '--dangerously-skip-permissions']
	assert '--model' in captured['argv']


@pytest.mark.asyncio
async def test_bcode_agent_enriches_result_from_export_transcript(tmp_path, monkeypatch):
	fake = tmp_path / 'bcode'
	png_b64 = base64.b64encode(b'fake-png').decode()
	fake.write_text(
		f"""#!/usr/bin/env python3
import json, sys
if sys.argv[1] == "run":
    print(json.dumps({{"type": "step_start", "timestamp": 1, "sessionID": "sess-export", "part": {{"id": "s1"}}}}), flush=True)
    raise SystemExit(0)
if sys.argv[1] == "export":
    print(json.dumps({{
        "info": {{
            "id": "sess-export",
            "cost": 0.25,
            "tokens": {{"input": 12, "output": 7, "reasoning": 0, "cache": {{"read": 0, "write": 0}}}}
        }},
        "messages": [{{
            "info": {{
                "role": "assistant",
                "providerID": "openai",
                "modelID": "gpt-5.5",
                "cost": 0.25,
                "tokens": {{"input": 12, "output": 7, "reasoning": 0, "cache": {{"read": 0, "write": 0}}}}
            }},
            "parts": [
                {{"type": "step-start", "id": "prt-start"}},
                {{"type": "tool", "id": "prt-tool", "tool": "browser_execute", "state": {{
                    "status": "completed",
                    "input": {{"code": "await session.Page.captureScreenshot({{format: 'png'}})"}},
                    "output": "saw page",
                    "title": "browser_execute",
                    "metadata": {{}},
                    "time": {{"start": 1000, "end": 1500}},
                    "attachments": [{{"type": "file", "mime": "image/png", "url": "data:image/png;base64,{png_b64}"}}]
                }}}},
                {{"type": "text", "id": "prt-text", "text": "Exported final answer"}},
                {{"type": "step-finish", "id": "prt-finish", "cost": 0.25, "tokens": {{"input": 12, "output": 7, "reasoning": 0, "cache": {{"read": 0, "write": 0}}}}}}
            ]
        }}]
    }}))
""",
	)
	fake.chmod(0o755)
	monkeypatch.setenv('BROWSER_USE_BCODE_BINARY', str(fake))

	result = await Agent(
		task='Use a browser',
		llm=_llm_class('ChatOpenAI')(model='gpt-5.5'),
		browser=_Browser(),
		workspace_dir=tmp_path / 'workspace',
	).run()

	assert result.final_result() == 'Exported final answer'
	assert result.usage.input_tokens == 12
	assert result.usage.output_tokens == 7
	assert result.usage.cost == 0.25
	assert result.steps[0].tool == 'browser_execute'
	assert result.steps[0].tool_output['output'] == 'saw page'
	assert result.steps[0].screenshot_paths
	assert result.history[0].state.get_screenshot() == png_b64
