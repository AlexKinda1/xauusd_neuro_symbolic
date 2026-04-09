"""
Definition of AI agent roles.
"""

from crewai import Agent

economist = Agent(
    role="Economist",
    goal="Analyze macroeconomic factors",
    backstory="Expert in commodity markets"
)

quant = Agent(
    role="Quantitative Analyst",
    goal="Provide model predictions",
    backstory="Time series forecasting expert"
)

risk_manager = Agent(
    role="Risk Manager",
    goal="Manage trading risks",
    backstory="Risk management specialist"
)