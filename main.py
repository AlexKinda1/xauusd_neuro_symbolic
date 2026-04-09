"""
Main entry point for the XAUUSD Neuro Symbolic Trading System.

Launches the Streamlit dashboard.
"""

import streamlit as st

from src.ui.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()