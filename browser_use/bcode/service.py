"""
Python `Agent` wrapper for BrowserCode/Bcode.

This is intentionally modeled after `browser_use.rust.Agent`: the Browser Use
library owns the public Python interface and the browser lifecycle, while Bcode
owns the inner agent/tool loop and attaches to the browser over CDP.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import shlex
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
BCODE_COMMAND_ENV = 'BROWSER_USE_BCODE_COMMAND'
BCODE_LEGACY_BINARY_ENVS = ('BCODE_BINARY', 'BCODE_BIN_PATH')
BCODE_BROWSER_EXECUTE_MAX_TIMEOUT_ENV = 'BROWSER_USE_BCODE_BROWSER_EXECUTE_MAX_TIMEOUT_MS'
BCODE_DEFAULT_BROWSER_EXECUTE_MAX_TIMEOUT_MS = 45_000
BCODE_CDP_CONNECT_TIMEOUT_ENV = 'BROWSER_USE_BCODE_CDP_CONNECT_TIMEOUT_MS'
BCODE_DEFAULT_CDP_CONNECT_TIMEOUT_MS = 30_000
BCODE_TOOL_OUTPUT_MAX_LINES = 600
BCODE_TOOL_OUTPUT_MAX_BYTES = 24_000


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

	raise BcodeNotInstalledError(
		'Bcode CLI not found. Install `bcode`, set BROWSER_USE_BCODE_BINARY to the executable path, '
		'or set BROWSER_USE_BCODE_COMMAND to a full command.'
	)


def _bcode_command() -> list[str]:
	command = os.environ.get(BCODE_COMMAND_ENV)
	if command:
		argv = shlex.split(command)
		if argv:
			return argv
		raise BcodeNotInstalledError(f'{BCODE_COMMAND_ENV} is empty after shell parsing.')
	return [str(find_bcode_binary())]


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
		controller = _unsupported.pop('controller', None)
		tools = _unsupported.pop('tools', None)
		self.output_model = (
			output_model
			or _unsupported.pop('output_model_schema', None)
			or _output_model_from_tools(controller)
			or _output_model_from_tools(tools)
		)
		self.state_dir = Path(state_dir) if state_dir else None
		self.workspace_dir = Path(workspace_dir) if workspace_dir else None
		self.extra_args = list(extra_args or [])
		ctor_max_steps = _unsupported.pop('max_steps', None) or _unsupported.pop('max_turns', None)
		self._ctor_max_steps: int | None = int(ctor_max_steps) if ctor_max_steps else None
		self.initial_actions = _unsupported.pop('initial_actions', None)
		self.override_system_message = _unsupported.pop('override_system_message', None)
		self.extend_system_message = _unsupported.pop('extend_system_message', None)
		self.sensitive_data = _unsupported.pop('sensitive_data', None)
		self.available_file_paths = list(_unsupported.pop('available_file_paths', None) or [])
		self.allowed_domains = _unsupported.pop('allowed_domains', None)
		self.blocked_domains = _unsupported.pop('blocked_domains', None)

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

	async def _judge_and_log(self) -> None:
		"""Run Browser Use's ComprehensiveV1 judge and store the verdict.

		The eval harness calls this method directly for local Browser Use agents.
		Expose the same compatibility hook as `browser_use.rust.Agent` so Bcode
		runs can be judged and saved through the standard pipeline.
		"""
		result = self.result
		if result is None or result.exit_code != 0:
			return

		try:
			from browser_use.agent.judge import construct_judge_messages
			from browser_use.agent.views import JudgementResult
		except Exception:
			return

		llm = _resolve_judge_llm()
		if llm is None:
			import logging

			logging.getLogger('browser_use.bcode.Agent').warning(
				'Judge LLM unavailable (no GEMINI_API_KEY / GOOGLE_API_KEY and no OPENAI_API_KEY) — skipping ComprehensiveV1 judge.'
			)
			return

		task = self.task or ''
		final_result = result.final_result() or ''
		agent_steps: list[str] = []
		for step in result.steps:
			tool = step.tool or '?'
			arg_keys = ','.join(sorted((step.tool_input or {}).keys()))
			out_keys = ','.join(sorted((step.tool_output or {}).keys()))
			agent_steps.append(f'{tool}(args={arg_keys}) -> ({out_keys})')

		screenshot_paths = [p for s in result.steps for p in s.screenshot_paths if p]

		try:
			messages = construct_judge_messages(
				task=task,
				final_result=final_result,
				agent_steps=agent_steps,
				screenshot_paths=screenshot_paths,
				max_images=10,
				ground_truth=None,
				use_vision=True,
			)
			response = await llm.ainvoke(messages, output_format=JudgementResult)
			judgement: JudgementResult = response.completion  # type: ignore[assignment]
		except Exception as exc:
			import logging

			logging.getLogger('browser_use.bcode.Agent').warning('Judge LLM call failed: %s', exc, exc_info=True)
			return

		result.judgement_dict = {
			'verdict': bool(judgement.verdict),
			'reasoning': judgement.reasoning or '',
			'failure_reason': judgement.failure_reason or '',
			'impossible_task': bool(judgement.impossible_task),
			'reached_captcha': bool(judgement.reached_captcha),
		}

	async def _run_headless(
		self,
		task: str,
		*,
		max_steps: int | None,
		on_step_start: Any | None,
		on_step_end: Any | None,
	) -> AgentRunResult:
		bcode_cmd = _bcode_command()
		started = time.monotonic()
		steps: list[StepRecord] = []
		stderr_chunks: list[bytes] = []
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
			await _execute_initial_actions(self.browser, self.initial_actions)
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
			screenshot_dir = bcode_dir / 'screenshots'
			instructions_file = self._write_browser_use_instructions(bcode_dir)
			_write_bcode_config(bcode_dir / 'bcode.json', self._model_id(), instructions_file, max_steps=max_steps)

			env = {**os.environ, **self._env_overrides(cdp_url, workspace, screenshot_dir=screenshot_dir)}
			argv = self._argv(bcode_cmd, task, max_steps=max_steps)

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
				buffer = b''
				while True:
					chunk = await proc.stdout.read(8192)
					if not chunk:
						break
					buffer += chunk
					while b'\n' in buffer:
						line, buffer = buffer.split(b'\n', 1)
						await apply_stdout_line(line)
					if len(buffer) > 100_000_000:
						failure = 'Bcode emitted a JSONL stdout event larger than 100MB.'
						buffer = b''
				if buffer.strip():
					await apply_stdout_line(buffer)

			async def apply_stdout_line(line: bytes) -> None:
				nonlocal final_summary, failure
				event = _parse_json_line(line)
				if event is None:
					return
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

			async def read_stderr() -> None:
				assert proc.stderr is not None
				while True:
					chunk = await proc.stderr.read(8192)
					if not chunk:
						break
					stderr_chunks.append(chunk)
					if sum(len(item) for item in stderr_chunks) > 1_000_000:
						stderr_chunks[:] = [b''.join(stderr_chunks)[-1_000_000:]]

			stdout_task = asyncio.create_task(read_stdout())
			stderr_task = asyncio.create_task(read_stderr())
			try:
				if self.timeout:
					exit_code = await asyncio.wait_for(proc.wait(), timeout=self.timeout)
				else:
					exit_code = await proc.wait()
			except asyncio.TimeoutError:
				await self.cancel()
				exit_code = 124
			except asyncio.CancelledError:
				await self.cancel()
				raise
			finally:
				with contextlib.suppress(Exception):
					await stdout_task
				with contextlib.suppress(Exception):
					await stderr_task

			export_data: dict[str, Any] | None = None
			export_usage: _UsageView | None = None
			if self.session_id:
				export_data = await _export_session(bcode_cmd, self.session_id, workspace, env)
				if export_data:
					export_steps, export_summary, export_usage = _steps_from_export(export_data, screenshot_dir)
					if export_steps:
						steps = export_steps
					final_summary = export_summary or final_summary

			result = AgentRunResult(
				session_id=self.session_id,
				exit_code=exit_code,
				final_summary=final_summary,
				failure=failure,
				steps=steps,
				events=[],
				stderr=b''.join(stderr_chunks).decode(errors='replace'),
				duration_seconds=time.monotonic() - started,
			)
			if self.output_model is not None and final_summary:
				result.final_output = _parse_output_model(self.output_model, final_summary)
			usage = export_usage if export_usage is not None else _UsageView()
			usage.model = usage.model or self._model_id()
			object.__setattr__(result, '_usage_cache', usage)
			return result
		finally:
			self._proc = None
			if owned_browser:
				with contextlib.suppress(Exception):
					await self.browser.stop()
			if workspace_cm is not None:
				workspace_cm.cleanup()

	def _argv(self, bcode_cmd: list[str], task: str, *, max_steps: int | None) -> list[str]:
		argv = [
			*bcode_cmd,
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
		for file_path in self.available_file_paths:
			path = Path(file_path).expanduser()
			if path.exists():
				argv.extend(['--file', str(path)])
			else:
				warnings.warn(f'browser_use.bcode.Agent available_file_paths entry does not exist: {path}', stacklevel=2)
		argv.append(task)
		return argv

	def _write_browser_use_instructions(self, bcode_dir: Path) -> Path | None:
		lines = _browser_use_instruction_lines(
			output_model=self.output_model,
			override_system_message=self.override_system_message,
			extend_system_message=self.extend_system_message,
			sensitive_data=self.sensitive_data,
			available_file_paths=self.available_file_paths,
			allowed_domains=self.allowed_domains,
			blocked_domains=self.blocked_domains,
		)
		if not lines:
			return None
		path = bcode_dir / 'browser-use-instructions.md'
		path.write_text('\n'.join(lines).strip() + '\n')
		return path

	def _model_id(self) -> str:
		model = self._model or 'gpt-5.5'
		if '/' in model:
			return model
		return f'{self.provider}/{model}'

	def _env_overrides(
		self,
		cdp_url: str | None = None,
		workspace: Path | None = None,
		*,
		screenshot_dir: Path | None = None,
	) -> dict[str, str]:
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
			env['BCODE_BROWSER_EXECUTE_MAX_TIMEOUT_MS'] = os.environ.get(
				BCODE_BROWSER_EXECUTE_MAX_TIMEOUT_ENV,
				str(BCODE_DEFAULT_BROWSER_EXECUTE_MAX_TIMEOUT_MS),
			)
			env['BCODE_CDP_CONNECT_TIMEOUT_MS'] = os.environ.get(
				BCODE_CDP_CONNECT_TIMEOUT_ENV,
				str(BCODE_DEFAULT_CDP_CONNECT_TIMEOUT_MS),
			)
		if screenshot_dir is not None:
			env['BCODE_SCREENSHOT_DIR'] = str(screenshot_dir)
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


async def _export_session(
	bcode_cmd: list[str],
	session_id: str,
	workspace: Path,
	env: dict[str, str],
) -> dict[str, Any] | None:
	proc = await asyncio.create_subprocess_exec(
		*bcode_cmd,
		'export',
		session_id,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
		cwd=str(workspace),
		env=env,
	)
	stdout, _stderr = await proc.communicate()
	if proc.returncode != 0 or not stdout:
		return None
	try:
		data = json.loads(stdout.decode(errors='replace'))
	except json.JSONDecodeError:
		return None
	return data if isinstance(data, dict) else None


def _steps_from_export(data: dict[str, Any], screenshot_dir: Path) -> tuple[list[StepRecord], str | None, _UsageView | None]:
	steps: list[StepRecord] = []
	final_summary: str | None = None
	usage = _usage_from_export(data)
	for msg in data.get('messages') or []:
		if not isinstance(msg, dict):
			continue
		info = msg.get('info') if isinstance(msg.get('info'), dict) else {}
		parts = msg.get('parts') if isinstance(msg.get('parts'), list) else []
		if info.get('role') != 'assistant':
			continue
		message_text_parts: list[str] = []
		current: StepRecord | None = None
		for part in parts:
			if not isinstance(part, dict):
				continue
			part_type = part.get('type')
			if part_type == 'step-start':
				current = StepRecord(seq=len(steps) + 1)
				steps.append(current)
				continue
			if current is None:
				current = StepRecord(seq=len(steps) + 1)
				steps.append(current)
			if part_type == 'text':
				text = part.get('text')
				if isinstance(text, str) and text.strip():
					message_text_parts.append(text)
					current.model_text = (current.model_text + '\n' + text).strip() if current.model_text else text
				continue
			if part_type == 'tool':
				_apply_export_tool_part(current, part, screenshot_dir)
				continue
			if part_type == 'step-finish':
				_apply_export_step_finish(current, part)
				continue
		if message_text_parts:
			final_summary = '\n'.join(message_text_parts).strip()
		if isinstance(info.get('structured'), (dict, list)):
			final_summary = json.dumps(info['structured'])
	return steps, final_summary, usage


def _apply_export_tool_part(step: StepRecord, part: dict[str, Any], screenshot_dir: Path) -> None:
	state = part.get('state') if isinstance(part.get('state'), dict) else {}
	step.tool = str(part.get('tool') or '')
	input_data = state.get('input')
	if isinstance(input_data, dict):
		step.tool_input = input_data
	output: dict[str, Any] = {
		'status': state.get('status'),
		'title': state.get('title'),
		'output': state.get('output'),
		'error': state.get('error'),
		'attachments': state.get('attachments'),
		'metadata': state.get('metadata'),
	}
	step.tool_output = {k: v for k, v in output.items() if v is not None}
	time_info = state.get('time') if isinstance(state.get('time'), dict) else {}
	if isinstance(time_info.get('start'), int):
		step.started_ts_ms = time_info['start']
	if isinstance(time_info.get('end'), int):
		step.finished_ts_ms = time_info['end']
	for idx, attachment in enumerate(state.get('attachments') or []):
		if not isinstance(attachment, dict):
			continue
		path = _materialize_attachment(attachment, screenshot_dir, f'{part.get("id") or step.seq}-{idx}')
		if path:
			step.screenshot_paths.append(path)


def _apply_export_step_finish(step: StepRecord, part: dict[str, Any]) -> None:
	tokens = part.get('tokens') if isinstance(part.get('tokens'), dict) else {}
	input_tokens = tokens.get('input')
	if isinstance(input_tokens, (int, float)):
		step.input_tokens = int(input_tokens)


def _usage_from_export(data: dict[str, Any]) -> _UsageView | None:
	info = data.get('info') if isinstance(data.get('info'), dict) else {}
	usage = _UsageView()
	tokens = info.get('tokens') if isinstance(info.get('tokens'), dict) else {}
	if tokens:
		usage.input_tokens = int(tokens.get('input') or 0)
		usage.output_tokens = int(tokens.get('output') or 0)
	cost = info.get('cost')
	if isinstance(cost, (int, float)):
		usage.cost = float(cost)
	if usage.input_tokens or usage.output_tokens or usage.cost:
		return usage

	for msg in data.get('messages') or []:
		if not isinstance(msg, dict):
			continue
		msg_info = msg.get('info') if isinstance(msg.get('info'), dict) else {}
		if msg_info.get('role') != 'assistant':
			continue
		msg_tokens = msg_info.get('tokens') if isinstance(msg_info.get('tokens'), dict) else {}
		usage.input_tokens += int(msg_tokens.get('input') or 0)
		usage.output_tokens += int(msg_tokens.get('output') or 0)
		msg_cost = msg_info.get('cost')
		if isinstance(msg_cost, (int, float)):
			usage.cost += float(msg_cost)
		if isinstance(msg_info.get('modelID'), str) and isinstance(msg_info.get('providerID'), str):
			usage.model = f'{msg_info["providerID"]}/{msg_info["modelID"]}'
	return usage if usage.input_tokens or usage.output_tokens or usage.cost else None


def _materialize_attachment(attachment: dict[str, Any], screenshot_dir: Path, name_hint: str) -> str | None:
	mime = attachment.get('mime')
	url = attachment.get('url')
	if not (isinstance(mime, str) and mime.startswith('image/') and isinstance(url, str)):
		return None
	if url.startswith('data:'):
		prefix, sep, payload = url.partition(',')
		if not sep or ';base64' not in prefix:
			return None
		ext = {
			'image/png': 'png',
			'image/jpeg': 'jpg',
			'image/webp': 'webp',
		}.get(mime, 'img')
		try:
			raw = base64.b64decode(payload, validate=True)
		except Exception:
			return None
		screenshot_dir.mkdir(parents=True, exist_ok=True)
		path = screenshot_dir / f'{_safe_filename(name_hint)}.{ext}'
		path.write_bytes(raw)
		return str(path)
	if url.startswith('file://'):
		return url.removeprefix('file://')
	return None


def _safe_filename(value: str) -> str:
	return ''.join(c if c.isalnum() or c in '-_.' else '-' for c in value)[:80] or 'attachment'


async def _maybe_call(fn: Any | None, *args: Any) -> None:
	if fn is None:
		return
	value = fn(*args)
	if hasattr(value, '__await__'):
		await value


def _write_bcode_config(
	path: Path,
	model_id: str,
	instructions_file: Path | None = None,
	*,
	max_steps: int | None = None,
) -> None:
	payload = {
		'model': model_id,
		'share': 'disabled',
		'autoupdate': False,
		'snapshot': False,
		'tool_output': {
			'max_lines': BCODE_TOOL_OUTPUT_MAX_LINES,
			'max_bytes': BCODE_TOOL_OUTPUT_MAX_BYTES,
		},
	}
	if instructions_file is not None:
		payload['instructions'] = [str(instructions_file)]
	if max_steps is not None:
		payload['agent'] = {'build': {'steps': max(1, int(max_steps))}}
	path.write_text(json.dumps(payload, indent=2) + '\n')


def _browser_use_instruction_lines(
	*,
	output_model: type[BaseModel] | None,
	override_system_message: Any,
	extend_system_message: Any,
	sensitive_data: Any,
	available_file_paths: list[Any],
	allowed_domains: Any,
	blocked_domains: Any,
) -> list[str]:
	lines: list[str] = [
		'# Browser Use run contract',
		'Complete the user task autonomously. Do not ask follow-up questions or offer next-step options in the final answer.',
		'When details are underspecified, make a reasonable assumption, state it briefly, and continue.',
		'If a source is blocked or unavailable, use the best available official or directly relevant alternative and clearly mark any uncertainty.',
		'Return the requested answer format as completely as possible within the run budget.',
		'',
	]
	if isinstance(override_system_message, str) and override_system_message.strip():
		lines.extend(['# Browser Use system override', override_system_message.strip(), ''])
	if isinstance(extend_system_message, str) and extend_system_message.strip():
		lines.extend(['# Browser Use additional system message', extend_system_message.strip(), ''])
	if output_model is not None:
		lines.extend(
			[
				'# Browser Use structured output',
				'When producing the final answer, output JSON matching this schema and no surrounding prose:',
				json.dumps(output_model.model_json_schema(), indent=2, sort_keys=True),
				'',
			]
		)
	if sensitive_data:
		lines.extend(
			[
				'# Browser Use sensitive data',
				'Use these sensitive values only when needed for the task. Do not reveal them in the final answer unless explicitly requested.',
				_json_or_text(sensitive_data),
				'',
			]
		)
	if available_file_paths:
		lines.extend(
			[
				'# Browser Use available files',
				'The Browser Use caller made these files available for this run. They are attached to the Bcode prompt when they exist:',
				_json_or_text([str(Path(path).expanduser()) for path in available_file_paths]),
				'',
			]
		)
	if allowed_domains:
		lines.extend(['# Browser Use allowed domains', _json_or_text(allowed_domains), ''])
	if blocked_domains:
		lines.extend(['# Browser Use blocked domains', _json_or_text(blocked_domains), ''])
	return lines


def _json_or_text(value: Any) -> str:
	try:
		return json.dumps(value, indent=2, sort_keys=True, default=str)
	except TypeError:
		return str(value)


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


async def _execute_initial_actions(browser: Any, initial_actions: Any) -> None:
	if not initial_actions:
		return
	for action in initial_actions:
		if not isinstance(action, dict):
			warnings.warn(
				f'browser_use.bcode.Agent only supports dict initial actions; ignored {type(action).__name__}.',
				stacklevel=2,
			)
			continue
		navigate = action.get('navigate')
		if not isinstance(navigate, dict) or not isinstance(navigate.get('url'), str):
			warnings.warn(
				f'browser_use.bcode.Agent only supports navigate initial actions; ignored {action!r}.',
				stacklevel=2,
			)
			continue
		await _navigate_browser(browser, navigate['url'], bool(navigate.get('new_tab', False)))


async def _navigate_browser(browser: Any, url: str, new_tab: bool) -> None:
	navigate_to = getattr(browser, 'navigate_to', None)
	if navigate_to is not None:
		value = navigate_to(url, new_tab=new_tab)
		if hasattr(value, '__await__'):
			await value
		return

	event_bus = getattr(browser, 'event_bus', None)
	dispatch = getattr(event_bus, 'dispatch', None)
	if dispatch is not None:
		from browser_use.browser.events import NavigateToUrlEvent

		value = dispatch(NavigateToUrlEvent(url=url, new_tab=new_tab))
		if hasattr(value, '__await__'):
			await value
		return

	warnings.warn(
		'browser_use.bcode.Agent received navigate initial_actions, but the browser does not support navigate_to().',
		stacklevel=2,
	)


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


def _output_model_from_tools(value: Any) -> type[BaseModel] | None:
	if value is None:
		return None
	get_output_model = getattr(value, 'get_output_model', None)
	if callable(get_output_model):
		output_model = get_output_model()
		if isinstance(output_model, type) and issubclass(output_model, BaseModel):
			return output_model
	output_model = getattr(value, '_output_model', None)
	if isinstance(output_model, type) and issubclass(output_model, BaseModel):
		return output_model
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


def _resolve_judge_llm() -> Any | None:
	"""Pick a cheap, independent judge LLM. Mirrors the Rust wrapper."""
	if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
		try:
			from browser_use.llm.google.chat import ChatGoogle

			return ChatGoogle(model='gemini-3-flash-preview')
		except Exception:
			pass

	if os.environ.get('OPENAI_API_KEY'):
		try:
			from browser_use.llm.openai.chat import ChatOpenAI

			return ChatOpenAI(model='gpt-4o-mini')
		except Exception:
			pass

	return None


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
