"""
Day 03 - ReAct Agent Implementation
Demonstrates the classic Reasoning + Acting loop (Yao et al., 2022) with scratchpad tracking.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from tools import Tool, get_default_tools

REACT_SYSTEM_PROMPT = """You are a reasoning assistant designed to solve problems using the ReAct (Reason + Act) framework.
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to {max_steps} times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

class ReActAgent:
    def __init__(self, tools: Optional[Dict[str, Tool]] = None, max_steps: int = 5):
        self.tools = tools or get_default_tools()
        self.max_steps = max_steps
        self.trajectory: List[Dict[str, Any]] = []

    def build_prompt(self, question: str) -> str:
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
        tool_names = ", ".join(self.tools.keys())
        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            max_steps=self.max_steps
        ) + f"\nQuestion: {question}\n"

    def parse_action(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses Thought, Action, and Action Input from agent output.
        Also checks for 'Final Answer:'.
        """
        if "Final Answer:" in text:
            parts = text.split("Final Answer:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            final_answer = parts[1].strip()
            return thought, "Final Answer", final_answer

        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=\nAction Input:|$)", text)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?=\nObservation:|$)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None
        action_input = action_input_match.group(1).strip() if action_input_match else None

        return thought, action, action_input

    def execute_step(self, action: str, action_input: str) -> str:
        """Invokes the selected tool and returns the observation string."""
        if action not in self.tools:
            return f"Error: Tool '{action}' is not valid. Choose one of: {list(self.tools.keys())}"
        
        tool = self.tools[action]
        return tool.execute(action_input)

    def run_simulated_trajectory(self, question: str, llm_step_generator) -> Dict[str, Any]:
        """
        Executes a complete ReAct loop given a step generator (simulating or wrapping an LLM).
        """
        scratchpad = f"Question: {question}\n"
        print(f"\n==========================================")
        print(f"[Objective] {question}")
        print(f"==========================================")

        for step in range(1, self.max_steps + 1):
            print(f"\n[Step {step}/{self.max_steps}] Generating Thought & Action...")
            step_output = llm_step_generator(scratchpad, step)
            
            thought, action, action_input = self.parse_action(step_output)
            print(f"  [Thought] {thought}")

            if action == "Final Answer":
                print(f"  [Final Answer] {action_input}")
                self.trajectory.append({"step": step, "thought": thought, "final_answer": action_input})
                return {
                    "status": "success",
                    "steps": step,
                    "final_answer": action_input,
                    "trajectory": self.trajectory
                }

            if not action or action_input is None:
                print("  [Warning] Could not parse valid Action and Action Input. Halting.")
                break

            print(f"  [Action] {action}({action_input})")
            observation = self.execute_step(action, action_input)
            print(f"  [Observation] {observation}")

            self.trajectory.append({
                "step": step,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation
            })

            scratchpad += f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"

        return {
            "status": "max_steps_exceeded",
            "steps": self.max_steps,
            "final_answer": None,
            "trajectory": self.trajectory
        }
