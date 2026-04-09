"""
Tool for risk calculations.
"""

class RiskTool:
    def calculate_stop_loss(self, entry_price, risk_percentage):
        return entry_price * (1 - risk_percentage)

    def calculate_take_profit(self, entry_price, reward_ratio):
        return entry_price * (1 + reward_ratio)