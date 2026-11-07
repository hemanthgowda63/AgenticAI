"""
Day 03 - ReAct Agent Tools
Provides executable tools for the ReAct Agent runtime.
"""
import ast
import operator
from typing import Any, Callable, Dict

class Tool:
    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, argument: str) -> str:
        try:
            return str(self.func(argument.strip()))
        except Exception as e:
            return f"Error executing {self.name}: {e}"


def safe_calculator(expr: str) -> str:
    """Safely evaluates basic arithmetic expressions."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in allowed_operators:
                raise ValueError(f"Unsupported operator: {op}")
            return allowed_operators[op](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in allowed_operators:
                raise ValueError(f"Unsupported unary operator: {op}")
            return allowed_operators[op](_eval(node.operand))
        else:
            raise TypeError(f"Unsupported syntax: {type(node)}")

    try:
        cleaned = expr.replace("^", "**")
        parsed = ast.parse(cleaned, mode="eval")
        return str(_eval(parsed))
    except Exception as err:
        return f"Calculation error: {err}"


def knowledge_search(query: str) -> str:
    """Knowledge base search for Agentic AI topics."""
    query_lower = query.lower()
    kb = {
        "react": "ReAct (Reasoning + Acting) was introduced by Yao et al. (2022). It interleaves chain-of-thought reasoning with domain-specific tool actions.",
        "langgraph": "LangGraph is a library by LangChain for building cyclical, stateful multi-agent workflows using directed graphs.",
        "crewai": "CrewAI is a role-playing autonomous agent framework where agents collaborate in hierarchical or sequential processes.",
        "autogen": "AutoGen is a Microsoft framework enabling multi-agent conversation systems with customizable and conversable agents.",
        "memory": "Agent memory is categorized into Working/Short-term Memory (context window, scratchpad) and Long-term Memory (vector search, episodic store).",
    }

    for key, info in kb.items():
        if key in query_lower:
            return info
    return f"No exact matches found for '{query}'. Available topics: {', '.join(kb.keys())}."


def get_default_tools() -> Dict[str, Tool]:
    """Returns a dict of standard tools for the ReAct agent."""
    return {
        "Calculator": Tool(
            name="Calculator",
            description="Evaluates mathematical expressions. Input must be a valid arithmetic string (e.g. '15 * 4 + 10').",
            func=safe_calculator
        ),
        "Search": Tool(
            name="Search",
            description="Searches an indexed knowledge base of AI & Agentic concepts. Input should be keywords.",
            func=knowledge_search
        ),
    }
