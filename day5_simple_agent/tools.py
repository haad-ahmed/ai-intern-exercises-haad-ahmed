"""AI Agent Tools - Calculator, Search, Weather, and Wikipedia.

This module provides tool implementations for the REACT agent.
"""

import math
import re
from typing import Union
from urllib.parse import quote

import requests

def evaluate_math_expression(expression: str) -> Union[int, float, str]:
    """Parse and evaluate a math expression.
    
    Removes common question phrases, then evaluates using Python's eval()
    with math module available. Returns int if result is whole number.
    """
    # Normalize expression
    expr = expression.lower().replace("^", "**")
    
    # Remove common question phrases
    expr = re.sub(
        r"\b what is\b|\b what's\b|\b calculate\b|\b compute\b|\b solve\b|\b evaluate\b|\b find the result of\b|\b how much is\b",
        "",
        expr,
    )
    expr = expr.replace("=", "").replace("?", "").strip()
    
    if not expr:
        return "Error: No expression provided"

    try:
        # Evaluate with math module and common functions available
        math_namespace = {"__builtins__": {}, "math": math}
        # Add common math functions for convenience
        for func in ["sqrt", "sin", "cos", "tan", "log", "exp"]:
            if hasattr(math, func):
                math_namespace[func] = getattr(math, func)
        
        result = eval(expr, math_namespace)
        
        # Return int if whole number
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as exc:
        return f"Error: {str(exc)}"


def calculator_tool(query: str) -> str:
    """Calculate a math expression from user query."""
    result = evaluate_math_expression(query)
    return f"Result: {result}"


def search_tool(query: str) -> str:
    """Search a knowledge base for relevant information."""
    knowledge = {
        "embeddings": "Embeddings are numerical vector representations of text.",
        "rag": "RAG stands for Retrieval Augmented Generation - helps LLMs avoid hallucinations.",
        "react": "ReAct is an agent framework combining Reasoning and Acting in a loop.",
    }
    key = query.lower()
    for k, v in knowledge.items():
        if k in key:
            return v
    return f"Search: '{query}' (general knowledge)"


def weather_tool(city: str) -> str:
    """Fetch current weather for a city using Open-Meteo API."""
    try:
        city_name = city.lower().strip()
        city_coords = {
            "lahore": (31.55, 74.35),
            "karachi": (24.86, 67.01),
            "islamabad": (33.68, 73.06),
            "tokyo": (35.68, 139.77),
            "london": (51.51, -0.13),
            "new york": (40.71, -74.01),
        }
        coords = city_coords.get(city_name, (31.55, 74.35))
        
        # Get weather data from API
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&current_weather=true"
        """using Open-Meteo API to get current weather for the city. If city not in predefined list, defaults to Lahore."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract and format weather
        temp = data["current_weather"]["temperature"]
        code = data["current_weather"]["weathercode"]
        condition = "Sunny" if code in [0, 1] else "Cloudy" if code in [2, 3] else "Rainy"
        return f"{temp}°C, {condition} in {city.title()}"
    except Exception:
        return f"Weather unavailable for {city}"


def wikipedia_tool(topic: str) -> str:
    """Fetch Wikipedia summary for a topic using Wikipedia REST API."""
    try:
        # Format topic for URL
        clean_topic = quote(topic.strip().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic}" 
        """using Wikipedia REST API to get summary for the topic. If topic not found, return appropriate message."""

        # Get summary
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", f"No summary for {topic}.")
        return f"No Wikipedia article for '{topic}'."
    except Exception:
        return f"Wikipedia unavailable for {topic}."

