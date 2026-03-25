"""
Digital Twin Simulation Module
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


class DigitalTwinSimulator:
    """Simulate transaction flows and fraud scenarios."""
    
    def __init__(self):
        """Initialize simulator."""
        self.normal_scenarios = []
        self.fraud_scenarios = []
        self.initialize_scenarios()
    
    def initialize_scenarios(self):
        """Initialize predefined scenarios."""
        # Normal scenarios
        self.normal_scenarios = [
            {
                "name": "Regular P2P Transfer",
                "description": "Normal peer-to-peer money transfer",
                "amount": np.random.uniform(500, 3000),
                "type": "P2P",
                "hour": np.random.choice([9, 10, 11, 14, 15, 18]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 0,
                "txn_count_24h": np.random.randint(1, 5),
            },
            {
                "name": "Bill Payment",
                "description": "Utility bill or phone recharge payment",
                "amount": np.random.uniform(500, 2000),
                "type": "Bill Payment",
                "hour": np.random.choice([10, 15, 19, 20]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 0,
                "txn_count_24h": np.random.randint(1, 3),
            },
            {
                "name": "ATM Withdrawal",
                "description": "Cash withdrawal from ATM",
                "amount": np.random.uniform(5000, 15000),
                "type": "Withdrawal",
                "hour": np.random.choice([10, 12, 15, 17]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 0,
                "txn_count_24h": np.random.randint(1, 4),
            }
        ]
        
        # Fraud scenarios
        self.fraud_scenarios = [
            {
                "name": "High Amount at Unusual Time",
                "description": "Large transfer at odd hours (2-4 AM)",
                "amount": np.random.uniform(35000, 50000),
                "type": "P2P",
                "hour": np.random.choice([1, 2, 3, 4]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 1,
                "txn_count_24h": np.random.randint(10, 25),
            },
            {
                "name": "Compromised New Device",
                "description": "First transaction from brand new device (hours old)",
                "amount": np.random.uniform(20000, 40000),
                "type": "P2P",
                "hour": np.random.choice([2, 3, 4, 23]),
                "device_age": np.random.uniform(0.1, 1),  # Hours to days
                "location_change": 1,
                "txn_count_24h": np.random.randint(1, 3),
            },
            {
                "name": "Rapid Succession Transactions",
                "description": "Multiple large transactions in short time from different locations",
                "amount": np.random.uniform(5000, 15000),
                "type": "P2P",
                "hour": np.random.choice([10, 11, 12, 13, 14]),
                "device_age": np.random.uniform(50, 730),
                "location_change": 1,
                "txn_count_24h": np.random.randint(15, 30),
            },
            {
                "name": "Location Anomaly",
                "description": "User jumping between distant locations within hours",
                "amount": np.random.uniform(10000, 25000),
                "type": "Merchant",
                "hour": np.random.choice([10, 12, 15]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 1,
                "txn_count_24h": np.random.randint(5, 15),
            },
            {
                "name": "Test Transaction Pattern",
                "description": "Small then large transactions (testing stolen card)",
                "amount": np.random.uniform(100, 500),  # Small test transaction
                "type": "Merchant",
                "hour": np.random.choice([2, 3, 4]),
                "device_age": np.random.uniform(100, 730),
                "location_change": 1,
                "txn_count_24h": 2,  # Few transactions
            }
        ]
    
    def get_normal_scenario(self, scenario_name=None):
        """
        Get a normal transaction scenario.
        
        Args:
            scenario_name: Specific scenario name or None for random
        
        Returns:
            dict: Transaction scenario
        """
        if scenario_name:
            scenario = next((s for s in self.normal_scenarios if s["name"] == scenario_name), None)
        else:
            scenario = np.random.choice(self.normal_scenarios)
        
        return scenario.copy() if scenario else self.normal_scenarios[0].copy()
    
    def get_fraud_scenario(self, scenario_name=None):
        """
        Get a fraud transaction scenario.
        
        Args:
            scenario_name: Specific scenario name or None for random
        
        Returns:
            dict: Transaction scenario
        """
        if scenario_name:
            scenario = next((s for s in self.fraud_scenarios if s["name"] == scenario_name), None)
        else:
            scenario = np.random.choice(self.fraud_scenarios)
        
        return scenario.copy() if scenario else self.fraud_scenarios[0].copy()
    
    def simulate_transaction_flow(self, scenario, base_sender="USER001", base_receiver="USER002"):
        """
        Simulate a complete transaction flow.
        
        Args:
            scenario: Transaction scenario dict
            base_sender: Base sender ID
            base_receiver: Base receiver ID
        
        Returns:
            dict: Simulated transaction
        """
        transaction = {
            "transaction_id": f"TXN{np.random.randint(1000000, 9999999)}",
            "sender": base_sender,
            "receiver": base_receiver,
            "amount": float(scenario["amount"]),
            "type": scenario["type"],
            "time": datetime.now() - timedelta(hours=np.random.randint(0, 24)),
            "hour": int(scenario["hour"]),
            "day_of_week": datetime.now().weekday(),
            "device_id": f"DEV{np.random.randint(100000, 999999)}",
            "sender_device_type": np.random.choice(["Android", "iOS", "Web"]),
            "receiver_device_type": np.random.choice(["Android", "iOS", "Web"]),
            "location": f"Location_{np.random.randint(1, 100)}",
            "device_age_days": float(scenario["device_age"]),
            "location_change_indicator": int(scenario["location_change"]),
            "transaction_count_24h": int(scenario["txn_count_24h"]),
            "scenario": scenario["name"],
            "description": scenario["description"]
        }
        
        return transaction
    
    def get_scenario_comparison(self):
        """
        Get comparison of normal vs fraud scenarios.
        
        Returns:
            dict: Comparison data
        """
        comparison = {
            "Normal Scenarios": self.normal_scenarios,
            "Fraud Scenarios": self.fraud_scenarios
        }
        return comparison
    
    @staticmethod
    def analyze_transaction_risk(transaction_dict):
        """
        Analyze transaction to identify risk factors.
        
        Args:
            transaction_dict: Transaction data
        
        Returns:
            dict: Risk analysis
        """
        risk_factors = []
        risk_score = 0
        
        # Check amount
        amount = transaction_dict.get("amount", 0)
        if amount > 40000:
            risk_factors.append(("High Amount", f"₹{amount:,.2f}"))
            risk_score += 20
        elif amount > 25000:
            risk_factors.append(("Medium-High Amount", f"₹{amount:,.2f}"))
            risk_score += 10
        
        # Check device age
        device_age = transaction_dict.get("device_age_days", 365)
        if device_age < 7:
            risk_factors.append(("Very New Device", f"{device_age:.1f} days old"))
            risk_score += 25
        elif device_age < 30:
            risk_factors.append(("New Device", f"{device_age:.1f} days old"))
            risk_score += 15
        
        # Check time
        hour = transaction_dict.get("hour", 12)
        if hour < 5 or hour >= 23:
            risk_factors.append(("Unusual Time", f"{hour}:00 (anomalous hours)"))
            risk_score += 15
        
        # Check location change
        loc_change = transaction_dict.get("location_change_indicator", 0)
        if loc_change > 0.5:
            risk_factors.append(("Location Change", "Rapid location shift"))
            risk_score += 20
        
        # Check transaction count
        txn_count = transaction_dict.get("transaction_count_24h", 0)
        if txn_count > 20:
            risk_factors.append(("High Frequency", f"{txn_count} transactions in 24h"))
            risk_score += 20
        elif txn_count > 10:
            risk_factors.append(("Medium Frequency", f"{txn_count} transactions in 24h"))
            risk_score += 10
        
        # Cap risk score
        risk_score = min(risk_score, 95)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "summary": f"Identified {len(risk_factors)} risk factor(s)"
        }
    
    def simulate_attack_scenario(self, attack_type="credential_compromise"):
        """
        Simulate different fraud attack scenarios.
        
        Args:
            attack_type: Type of attack
        
        Returns:
            list: Sequence of fraudulent transactions
        """
        attack_sequence = []
        base_time = datetime.now()
        
        if attack_type == "credential_compromise":
            # Attacker gains access and quickly drains account
            for i in range(5):
                transaction = {
                    "sequence": i + 1,
                    "time": base_time - timedelta(hours=5-i),
                    "amount": np.random.uniform(5000, 15000),
                    "location": f"Location_{np.random.randint(1, 100)}",
                    "type": "P2P",
                    "status": "FRAUDULENT"
                }
                attack_sequence.append(transaction)
        
        elif attack_type == "sim_swap":
            # SIM swap attack
            attack_sequence = [
                {
                    "sequence": 1,
                    "event": "Phone number ported to attacker's SIM",
                    "time": base_time - timedelta(hours=2),
                    "status": "COMPROMISED"
                },
                {
                    "sequence": 2,
                    "event": "Password reset via OTP",
                    "time": base_time - timedelta(hours=1.5),
                    "status": "COMPROMISED"
                },
                {
                    "sequence": 3,
                    "event": "Large withdrawal transaction",
                    "time": base_time - timedelta(hours=1),
                    "amount": 40000,
                    "status": "FRAUDULENT"
                }
            ]
        
        elif attack_type == "card_testing":
            # Test small transactions before larger fraud
            for i in range(3):
                amount = 100 + i * 50  # Increasing amounts
                transaction = {
                    "sequence": i + 1,
                    "amount": amount,
                    "time": base_time - timedelta(hours=3-i*0.5),
                    "type": "Merchant",
                    "status": "TESTING" if i < 2 else "FRAUDULENT"
                }
                attack_sequence.append(transaction)
        
        return attack_sequence


def get_digital_twin_dashboard_data():
    """
    Get comprehensive digital twin simulation data for dashboard.
    
    Returns:
        dict: All simulation scenarios and comparisons
    """
    simulator = DigitalTwinSimulator()
    
    # Generate sample transactions
    normal_txns = [
        simulator.simulate_transaction_flow(simulator.get_normal_scenario())
        for _ in range(5)
    ]
    
    fraud_txns = [
        simulator.simulate_transaction_flow(simulator.get_fraud_scenario())
        for _ in range(5)
    ]
    
    return {
        "normal_scenarios": simulator.normal_scenarios,
        "fraud_scenarios": simulator.fraud_scenarios,
        "sample_normal_transactions": normal_txns,
        "sample_fraud_transactions": fraud_txns,
        "scenario_comparison": simulator.get_scenario_comparison()
    }
