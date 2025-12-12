from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import re
from ..infra.openrouter.client import OpenRouterClient
from ..tools.python_repl import run_python_code
from ..tools.search import search_web

# Specialized system prompts for the Scientific Reasoning Agent
# Enhanced to handle current events and real-time information queries
SYSTEM_SCIENTIST = (
    "You are a Scientific Reasoning Agent equipped with tools.\n"
    "Your Goal: Solve problems by gathering evidence and verifying with tools.\n"
    "\n"
    "TOOLS AVAILABLE:\n"
    "1. `python_repl(code)`: Execute Python code. Use `print()` to see results. Pre-installed: numpy, sympy, networkx.\n"
    "2. `web_search(query)`: Search the web for current events, news, facts, and recent information.\n"
    "\n"
    "METHODOLOGY:\n"
    "- For questions about CURRENT EVENTS, PEOPLE'S STATUS, or RECENT NEWS -> ALWAYS SEARCH FIRST.\n"
    "- For arithmetic, counting, or simulation -> WRITE CODE.\n"
    "- NEVER rely solely on your training data for factual claims about:\n"
    "  * Whether someone is alive or dead\n"
    "  * Recent events (anything after 2023)\n"
    "  * Current positions/roles of people\n"
    "  * Breaking news or recent developments\n"
    "\n"
    "SEARCH TIPS:\n"
    "- When searching for person status (alive/dead), search: '<name> death 2024 2025' or '<name> news today'\n"
    "- For current events, include recent years in your query\n"
    "- Be specific: '<name> assassinated 2025' or '<name> current status'\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "To use a tool, output a block like:\n"
    "```search\n"
    "query string here\n"
    "```\n"
    "OR\n"
    "```python\n"
    "# code here\n"
    "print('result')\n"
    "```\n"
    "The system will intercept these blocks, run them, and append the output to your context.\n"
    "When you have the final answer, output: `FINAL ANSWER: <answer>`\n"
    "\n"
    "IMPORTANT: For questions about living people or recent events, you MUST search before answering."
)

class ScientistAgent:
    def __init__(
        self,
        client: OpenRouterClient,
        model_name: str,
        max_steps: int = 8
    ):
        self.client = client
        self.model_name = model_name
        self.max_steps = max_steps
        self._thinking_callback: Optional[Callable[[str, str], None]] = None

    def set_thinking_callback(self, callback: Callable[[str, str], None]) -> None:
        self._thinking_callback = callback

    def _emit(self, stage: str, content: str):
        if self._thinking_callback:
            self._thinking_callback(stage, content)

    def run(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_SCIENTIST},
            {"role": "user", "content": f"Solve this problem:\n{problem}"}
        ]
        
        for step in range(self.max_steps):
            self._emit("scientist_step", f"Step {step+1}/{self.max_steps}")
            
            # 1. Think
            result = self.client.complete_chat(messages=messages, model=self.model_name, temperature=0.0)
            response = result.text
            self._emit("scientist_thought", response)
            messages.append({"role": "assistant", "content": response})
            
            # 2. Check for Final Answer
            if "FINAL ANSWER:" in response:
                return response.split("FINAL ANSWER:")[1].strip()
            
            # 3. Check for Tools
            tool_output = ""
            
            # Parse Python
            py_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
            for code in py_blocks:
                self._emit("tool_exec", "Running Python code...")
                out = run_python_code(code)
                tool_output += f"\n[Python Output]:\n{out}\n"
                
            # Parse Search
            search_blocks = re.findall(r"```search\n(.*?)\n```", response, re.DOTALL)
            for query in search_blocks:
                self._emit("tool_exec", f"Searching web for: {query.strip()}...")
                out = search_web(query.strip())
                tool_output += f"\n[Search Output]:\n{out}\n"
                
            # 4. Observe
            if tool_output:
                self._emit("tool_result", tool_output[:500] + "..." if len(tool_output) > 500 else tool_output)
                messages.append({"role": "user", "content": f"Tool Observations:\n{tool_output}\n\nProceed with the next step based on these findings."})
            else:
                # If no tool used but no final answer, encourage tool use or conclusion
                messages.append({"role": "user", "content": "Please verify your reasoning using a tool (Python simulation or Search) if applicable, or state the Final Answer."})
                
        return "Unable to converge on an answer within the step limit."

