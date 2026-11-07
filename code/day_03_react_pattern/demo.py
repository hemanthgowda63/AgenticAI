"""
Day 03 - ReAct Agent Demo
Executes a multi-step reasoning task demonstrating the ReAct loop.
"""
from agent import ReActAgent
from tools import get_default_tools

def simulated_llm_steps(scratchpad: str, step: int) -> str:
    """
    Simulates LLM response generation based on the scratchpad state.
    Demonstrates reasoning, tool choice, observation digestion, and final answer synthesis.
    """
    if step == 1:
        return (
            "Thought: The user is asking what ReAct is and wants to calculate (25 * 4) + 15 to see if our calculator works.\n"
            "First, I will search our knowledge base for the definition of ReAct.\n"
            "Action: Search\n"
            "Action Input: react"
        )
    elif step == 2:
        return (
            "Thought: I now have the definition of ReAct. Next, I need to evaluate the math expression: (25 * 4) + 15.\n"
            "Action: Calculator\n"
            "Action Input: (25 * 4) + 15"
        )
    elif step == 3:
        return (
            "Thought: I have both the knowledge base result and the calculation result.\n"
            "Calculation result: 115.\n"
            "I have all required information to formulate the final answer.\n"
            "Final Answer: ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with tool execution (Yao et al., 2022). Additionally, the calculation (25 * 4) + 15 evaluates to 115."
        )
    return "Final Answer: Done."


def main():
    tools = get_default_tools()
    agent = ReActAgent(tools=tools, max_steps=5)

    prompt = agent.build_prompt("What is ReAct in Agentic AI, and what is (25 * 4) + 15?")
    print("Generated Agent System & Format Prompt:\n" + "-"*40)
    print(prompt)
    print("-" * 40)

    result = agent.run_simulated_trajectory(
        question="What is ReAct in Agentic AI, and what is (25 * 4) + 15?",
        llm_step_generator=simulated_llm_steps
    )

    print("\n==========================================")
    print("Execution Result Status:", result["status"])
    print("Completed in Steps:", result["steps"])
    print("Final Output:\n", result["final_answer"])
    print("==========================================")

if __name__ == "__main__":
    main()
