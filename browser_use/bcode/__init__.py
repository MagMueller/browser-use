"""
browser_use.bcode - Bcode-backed Browser Use agent.

Public entry point:

    from browser_use.bcode import Agent

    agent = Agent(task="Go to example.com and summarize it", llm=llm, browser=browser)
    history = await agent.run(max_steps=50)
"""

from browser_use.bcode.service import Agent, BcodeNotInstalledError, find_bcode_binary
from browser_use.bcode.views import AgentRunResult, StepRecord

__all__ = [
	'Agent',
	'AgentRunResult',
	'BcodeNotInstalledError',
	'StepRecord',
	'find_bcode_binary',
]
