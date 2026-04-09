"""
Crew orchestrator for managing AI agents.
"""

from crewai import Crew
from roles import economist, quant, risk_manager
from tasks import define_tasks

def run_crew():
    crew = Crew(
        agents=[economist, quant, risk_manager],
        tasks=define_tasks()
    )
    result = crew.kickoff()
    return result