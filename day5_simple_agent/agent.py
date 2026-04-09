"""
Day 5 : Building a Simple AI Agent with Gemini.

Features:
- Decides when to use tools
- Supports Calculator, Search, Wikipedia, and Weather
- Generates natural final answers
- Prints full REACT-style reasoning
"""

import os
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from tools import calculator_tool, search_tool, weather_tool, wikipedia_tool

class SimpleAgent:
    """REACT-style AI Agent with multiple tools."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        print("Agent is ready to Answer\n")

    def _decide_tool(self, query: str) -> str:
        """Decide which tool to use (or none)."""
        prompt = f"""You are a helpful AI agent.
User question: {query}

Available tools:
- calculator → use only for math calculations
- search → use for general knowledge, facts, explanations
- wikipedia → use for encyclopedia-style summaries
- weather → use for current weather queries

Respond with exactly one line:
TOOL: calculator
TOOL: search
TOOL: wikipedia
TOOL: weather
TOOL: none

Only choose calculator for clear math. Choose wikipedia for summaries, weather for current conditions, and search for general knowledge."""

        response = self.model.generate_content(prompt)
        decision = response.text.strip().lower()
        if "calculator" in decision:
            return "calculator"
        if "wikipedia" in decision:
            return "wikipedia"
        if "weather" in decision:
            return "weather"
        if "search" in decision:
            return "search"
        return "none"

    def _get_final_answer(self, query: str, tool_used: str, tool_result: Optional[str] = None) -> str:
        """Generate natural final answer."""
        if tool_used != "none" and tool_result:
            prompt = f"""User question: {query}
Tool used: {tool_used}
Tool result: {tool_result}

Write a clear, helpful, and natural final answer."""
        else:
            prompt = f"Answer this directly: {query}"

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def run(self, query: str) -> None:
        """Run the full REACT agent loop with visible reasoning."""
        print(f"🔍 Query: {query}")

        # Step 1: Thought
        print("Thought: Deciding what to do...")

        # Step 2: Action
        tool = self._decide_tool(query)
        print(f"Action: Using tool → {tool}")

        # Step 3: Observation
        if tool == "calculator":
            tool_result = calculator_tool(query)
        elif tool == "search":
            tool_result = search_tool(query)
        elif tool == "wikipedia":
            tool_result = wikipedia_tool(query)
        elif tool == "weather":
            tool_result = weather_tool(query)
        else:
            tool_result = None

        print(f"Observation: {tool_result if tool_result else 'No tool used'}")

        # Step 4: Final Answer
        answer = self._get_final_answer(query, tool, tool_result)
        print(f"\nFinal Answer: {answer}\n")
        print(" " * 70)


if __name__ == "__main__":
    agent = SimpleAgent()

    print("Type questions below (type 'admin123-says-exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"admin123-says-exit"}:
            print("Goodbye!")
            break
        if user_input:
            agent.run(user_input)
