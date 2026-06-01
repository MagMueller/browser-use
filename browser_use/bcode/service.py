"""
Python `Agent` wrapper for BrowserCode/Bcode.

This is intentionally modeled after `browser_use.rust.Agent`: the Browser Use
library owns the public Python interface and the browser lifecycle, while Bcode
owns the inner agent/tool loop and attaches to the browser over CDP.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from browser_use.bcode.views import AgentRunResult, StepRecord
from browser_use.rust.views import _UsageView

BCODE_BINARY_ENV = 'BROWSER_USE_BCODE_BINARY'
BCODE_LEGACY_BINARY_ENVS = ('BCODE_BINARY', 'BCODE_BIN_PATH')


class BcodeNotInstalledError(RuntimeError):
	"""Raised when the Bcode CLI cannot be found."""


OnEvent = Callable[[dict[str, Any]], None] | Callable[[dict[str, Any]], Awaitable[None]]


_PROVIDER_BY_CLASS: dict[str, str] = {
	'ChatOpenAI': 'openai',
	'ChatAzureOpenAI': 'openai',
	'ChatBrowserUse': 'openai',
	'ChatAnthropic': 'anthropic',
	'ChatGoogle': 'google',
	'ChatGemini': 'google',
	'ChatGroq': 'groq',
	'ChatOpenRouter': 'openrouter',
	'ChatDeepSeek': 'deepseek',
}

_PROVIDER_API_KEY_ENV: dict[str, str] = {
	'openai': 'OPENAI_API_KEY',
	'anthropic': 'ANTHROPIC_API_KEY',
	'google': 'GOOGLE_API_KEY',
	'groq': 'GROQ_API_KEY',
	'openrouter': 'OPENROUTER_API_KEY',
	'deepseek': 'DEEPSEEK_API_KEY',
}

_KNOWN_LEGACY_KWARGS: frozenset[str] = frozenset(
	{
		'browser_session',
		'output_model_schema',
		'max_steps',
		'max_turns',
		'controller',
		'tools',
		'use_vision',
		'max_actions_per_step',
		'use_thinking',
		'flash_mode',
		'images_per_step',
		'source',
		'calculate_cost',
		'override_system_message',
		'extend_system_message',
		'initial_actions',
		'use_judge',
		'judge_llm',
		'ground_truth',
		'browser_profile',
		'save_conversation_path',
		'max_failures',
		'skill_ids',
		'skills',
		'sensitive_data',
		'available_file_paths',
		'allowed_domains',
		'blocked_domains',
		'webhook_url',
		'webhook_token',
		'viewport_size',
		'window_size',
		'recordings_dir',
		'live_url_callback',
		'metadata',
		'tags',
		'name',
		'description',
		'priority',
		'budget_usd',
		'budget_tokens',
		'cache_key',
		'deterministic_replay',
	}
)


class _MessageManagerStub:
	def __init__(self) -> None:
		self.last_input_messages: list[Any] = []


def find_bcode_binary() -> Path:
	"""Find the Bcode executable.

	Resolution order is explicit env override, legacy Bcode env overrides,
	then PATH. Source checkouts are deliberately not auto-run because they need
	Bun/node dependencies; callers can point `BROWSER_USE_BCODE_BINARY` at a
	built binary or wrapper script.
	"""

	for key in (BCODE_BINARY_ENV, *BCODE_LEGACY_BINARY_ENVS):
		value = os.environ.get(key)
		if value:
			path = Path(value).expanduser()
			if path.exists():
				return path
			raise BcodeNotInstalledError(f'{key} points to missing Bcode binary: {path}')

	found = shutil.which('bcode')
	if found:
		return Path(found)

	raise BcodeNotInstalledError('Bcode CLI not found. Install `bcode` or set BROWSER_USE_BCODE_BINARY to the executable path.')


class Agent:
	def __init__(
		self,
		task: str | None = None,
		*,
		llm: Any | None = None,
		browser: Any | None = None,
		timeout: float | None = None,
		on_event: OnEvent | None = None,
		output_model: type[BaseModel] | None = None,
		state_dir: str | Path | None = None,
		workspace_dir: str | Path | None = None,
		extra_args: list[str] | None = None,
		**_unsupported: Any,
	) -> None:
		if browser is None and _unsupported.get('browser_session') is not None:
			browser = _unsupported.pop('browser_session')

		self.task = task
		self.llm = llm
		self.browser = browser
		self.timeout = timeout
		self.on_event = on_event
		self.output_model = output_model or _unsupported.pop('output_model_schema', None)
		self.state_dir = Path(state_dir) if state_dir else None
		self.workspace_dir = Path(workspace_dir) if workspace_dir else None
		self.extra_args = list(extra_args or [])
		ctor_max_steps = _unsupported.pop('max_steps', None) or _unsupported.pop('max_turns', None)
		self._ctor_max_steps: int | None = int(ctor_max_steps) if ctor_max_steps else None

		self.provider = _provider_from_llm(llm)
		self._model = _model_from_llm(llm)
		self._api_key = _api_key_from_llm(llm)
		self.session_id: str | None = None
		self.result: AgentRunResult | None = None
		self._proc: asyncio.subprocess.Process | None = None
		self.message_manager = _MessageManagerStub()

		unknown = {k: v for k, v in _unsupported.items() if k not in _KNOWN_LEGACY_KWARGS}
		if unknown:
			warnings.warn(
				f'browser_use.bcode.Agent does not honour kwargs: {", ".join(sorted(unknown))}. '
				'They were accepted for legacy compatibility but ignored.',
				stacklevel=2,
			)

	async def run(
		self,
		max_steps: int | None = None,
		*,
		on_step_start: Any | None = None,
		on_step_end: Any | None = None,
		**_unused: Any,
	) -> AgentRunResult:
		_ = _unused
		if self.task is None:
			raise ValueError('browser_use.bcode.Agent.run() requires a task.')

		effective_max_steps = max_steps or self._ctor_max_steps
		result = await self._run_headless(
			self.task,
			max_steps=effective_max_steps,
			on_step_start=on_step_start,
			on_step_end=on_step_end,
		)
		self.result = result
		return result

	async def cancel(self) -> None:
		proc = self._proc
		if proc and proc.returncode is None:
			proc.terminate()
			with contextlib.suppress(asyncio.TimeoutError):
				await asyncio.wait_for(proc.wait(), timeout=5)
			if proc.returncode is None:
				proc.kill()

	@property
	def history(self) -> Any:
		if self.result is not None:
			return self.result
		return AgentRunResult(exit_code=0, session_id=self.session_id)

	@property
	def usage(self) -> Any:
		if self.result is not None:
			return self.result.usage
		return _UsageView()

	async def _run_headless(
		self,
		task: str,
		*,
		max_steps: int | None,
		on_step_start: Any | None,
		on_step_end: Any | None,
	) -> AgentRunResult:
		cli = find_bcode_binary()
		started = time.monotonic()
		steps: list[StepRecord] = []
		stderr_blob = b''
		final_summary: str | None = None
		failure: str | None = None
		workspace_cm: tempfile.TemporaryDirectory[str] | None = None
		owned_browser = False

		try:
			if self.browser is None:
				from browser_use.browser import BrowserSession

				self.browser = BrowserSession(headless=True)
				owned_browser = True

			await _ensure_browser_started(self.browser)
			cdp_url = _browser_cdp_url(self.browser)
			if not cdp_url:
				raise RuntimeError('BrowserSession did not expose a CDP websocket URL after start().')

			if self.workspace_dir is None:
				workspace_cm = tempfile.TemporaryDirectory(prefix='browser-use-bcode-')
				workspace = Path(workspace_cm.name)
			else:
				workspace = self.workspace_dir
				workspace.mkdir(parents=True, exist_ok=True)

			bcode_dir = workspace / '.bcode'
			bcode_dir.mkdir(parents=True, exist_ok=True)
			_write_bcode_config(bcode_dir / 'bcode.json', self._model_id())

			env = {**os.environ, **self._env_overrides(cdp_url, workspace)}
			argv = self._argv(cli, task, max_steps=max_steps)

			proc = await asyncio.create_subprocess_exec(
				*argv,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(workspace),
				env=env,
			)
			self._proc = proc

			async def read_stdout() -> None:
				nonlocal final_summary, failure
				assert proc.stdout is not None
				while True:
					line = await proc.stdout.readline()
					if not line:
						break
					event = _parse_json_line(line)
					if event is None:
						continue
					await self._emit(event)
					next_summary, next_failure = await _apply_bcode_event(
						event,
						steps,
						on_step_start=on_step_start,
						on_step_end=on_step_end,
					)
					final_summary = next_summary or final_summary
					failure = next_failure or failure
					if event.get('sessionID') and not self.session_id:
						self.session_id = str(event['sessionID'])

			stdout_task = asyncio.create_task(read_stdout())
			try:
				if self.timeout:
					exit_code = await asyncio.wait_for(proc.wait(), timeout=self.timeout)
				else:
					exit_code = await proc.wait()
			except asyncio.TimeoutError:
				await self.cancel()
				exit_code = 124
			finally:
				await stdout_task
				if proc.stderr is not None:
					stderr_blob = await proc.stderr.read()

			result = AgentRunResult(
				session_id=self.session_id,
				exit_code=exit_code,
				final_summary=final_summary,
				failure=failure,
				steps=steps,
				events=[],
				stderr=stderr_blob.decode(errors='replace'),
				duration_seconds=time.monotonic() - started,
			)
			if self.output_model is not None and final_summary:
				result.final_output = _parse_output_model(self.output_model, final_summary)
			usage = _UsageView()
			usage.model = self._model_id()
			object.__setattr__(result, '_usage_cache', usage)
			return result
		finally:
			self._proc = None
			if owned_browser:
				with contextlib.suppress(Exception):
					await self.browser.stop()
			if workspace_cm is not None:
				workspace_cm.cleanup()

	def _argv(self, cli: Path, task: str, *, max_steps: int | None) -> list[str]:
		argv = [
			str(cli),
			'run',
			'--format',
			'json',
			'--dangerously-skip-permissions',
			'--model',
			self._model_id(),
		]
		if max_steps is not None:
			task = f'[Browser Use run budget: max_steps={int(max_steps)}]\n\n{task}'
		argv.extend(self.extra_args)
		argv.append(task)
		return argv

	def _model_id(self) -> str:
		model = self._model or 'gpt-5.5'
		if '/' in model:
			return model
		return f'{self.provider}/{model}'

	def _env_overrides(self, cdp_url: str | None = None, workspace: Path | None = None) -> dict[str, str]:
		env: dict[str, str] = {}
		if self._api_key:
			env[_PROVIDER_API_KEY_ENV.get(self.provider, 'OPENAI_API_KEY')] = self._api_key
		if cdp_url:
			env['BU_CDP_WS'] = cdp_url
			env['BU_CDP_URL'] = cdp_url
		if workspace is not None:
			base = self.state_dir or workspace / '.bcode-runtime'
			env['XDG_DATA_HOME'] = str(base / 'data')
			env['XDG_CONFIG_HOME'] = str(base / 'config')
			env['XDG_STATE_HOME'] = str(base / 'state')
			env['XDG_CACHE_HOME'] = str(base / 'cache')
			env['OPENCODE_CONFIG_DIR'] = str(workspace / '.bcode')
			env['OPENCODE_DISABLE_AUTOUPDATE'] = '1'
			env['OPENCODE_DISABLE_PROJECT_CONFIG'] = '0'
			env['OPENCODE_DISABLE_PRUNE'] = '1'
		return env

	async def _emit(self, event: dict[str, Any]) -> None:
		if self.on_event is None:
			return
		value = self.on_event(event)
		if hasattr(value, '__await__'):
			await value


async def _apply_bcode_event(
	event: dict[str, Any],
	steps: list[StepRecord],
	*,
	on_step_start: Any | None,
	on_step_end: Any | None,
) -> tuple[str | None, str | None]:
	event_type = event.get('type')
	part = event.get('part') if isinstance(event.get('part'), dict) else {}

	if event_type == 'step_start':
		step = StepRecord(seq=len(steps) + 1)
		steps.append(step)
		await _maybe_call(on_step_start, step)
		return None, None

	if event_type == 'step_finish':
		if not steps:
			steps.append(StepRecord(seq=1))
		await _maybe_call(on_step_end, steps[-1])
		return None, None

	if event_type == 'text':
		text = part.get('text')
		if isinstance(text, str) and text.strip():
			if not steps:
				steps.append(StepRecord(seq=1))
			steps[-1].model_text = text
			return text, None
		return None, None

	if event_type == 'tool_use':
		if not steps:
			steps.append(StepRecord(seq=1))
		tool_state = part.get('state') if isinstance(part.get('state'), dict) else {}
		steps[-1].tool = str(part.get('tool') or '')
		input_data = tool_state.get('input')
		if isinstance(input_data, dict):
			steps[-1].tool_input = input_data
		output = {
			'status': tool_state.get('status'),
			'title': tool_state.get('title'),
			'output': tool_state.get('output'),
			'error': tool_state.get('error'),
			'attachments': tool_state.get('attachments'),
		}
		steps[-1].tool_output = {k: v for k, v in output.items() if v is not None}
		return None, str(tool_state.get('error')) if tool_state.get('status') == 'error' else None

	if event_type == 'error':
		error = event.get('error')
		if isinstance(error, dict):
			return None, str(error.get('message') or error.get('name') or error)
		return None, str(error)

	return None, None


async def _maybe_call(fn: Any | None, *args: Any) -> None:
	if fn is None:
		return
	value = fn(*args)
	if hasattr(value, '__await__'):
		await value


def _write_bcode_config(path: Path, model_id: str) -> None:
	payload = {
		'model': model_id,
		'share': 'disabled',
		'autoupdate': False,
		'snapshot': False,
	}
	path.write_text(json.dumps(payload, indent=2) + '\n')


def _parse_json_line(line: bytes) -> dict[str, Any] | None:
	try:
		value = json.loads(line.decode(errors='replace'))
	except json.JSONDecodeError:
		return None
	return value if isinstance(value, dict) else None


async def _ensure_browser_started(browser: Any) -> None:
	if browser is None or _browser_cdp_url(browser):
		return
	start = getattr(browser, 'start', None)
	if start is None:
		return
	value = start()
	if hasattr(value, '__await__'):
		await value


def _provider_from_llm(llm: Any) -> str:
	if llm is None:
		return 'openai'
	return _PROVIDER_BY_CLASS.get(type(llm).__name__, 'openai')


def _model_from_llm(llm: Any) -> str | None:
	if llm is None:
		return None
	for attr in ('model', 'model_name', 'name'):
		value = getattr(llm, attr, None)
		if isinstance(value, str) and value:
			return value
	return None


def _api_key_from_llm(llm: Any) -> str | None:
	if llm is None:
		return None
	value = getattr(llm, 'api_key', None)
	if value is None:
		return None
	if hasattr(value, 'get_secret_value'):
		return value.get_secret_value()
	value = str(value)
	return value or None


def _browser_cdp_url(browser: Any) -> str | None:
	return _read_browser_attr(browser, ('cdp_url', 'wss_url'))


def _read_browser_attr(browser: Any, attr_names: tuple[str, ...]) -> Any:
	if browser is None:
		return None
	for attr in attr_names:
		direct = getattr(browser, attr, None)
		if direct is not None:
			return direct
	for nested_attr in ('browser_profile', 'profile'):
		nested = getattr(browser, nested_attr, None)
		if nested is None:
			continue
		for attr in attr_names:
			value = getattr(nested, attr, None)
			if value is not None:
				return value
	return None


def _parse_output_model(output_model: type[BaseModel], text: str) -> Any:
	try:
		return output_model.model_validate_json(text)
	except Exception:
		start = text.find('{')
		end = text.rfind('}')
		if start >= 0 and end > start:
			return output_model.model_validate_json(text[start : end + 1])
		raise
