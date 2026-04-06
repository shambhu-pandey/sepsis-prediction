"""
Digital Twin Simulation & Forecasting Module
Per-domain HMM trajectory simulation with DRL-style intervention.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─────────── Transition matrices per domain ──────────────────

DOMAIN_TRANSITIONS = {
    "paysim": {
        "Normal": {"Normal": 0.93, "Risk": 0.05, "Fraud": 0.02},
        "Risk":   {"Normal": 0.35, "Risk": 0.50, "Fraud": 0.15},
        "Fraud":  {"Normal": 0.05, "Risk": 0.15, "Fraud": 0.80},
    },
    "banksim": {
        "Normal": {"Normal": 0.95, "Risk": 0.04, "Fraud": 0.01},
        "Risk":   {"Normal": 0.40, "Risk": 0.45, "Fraud": 0.15},
        "Fraud":  {"Normal": 0.10, "Risk": 0.15, "Fraud": 0.75},
    },
    "cicids": {
        "Normal": {"Normal": 0.90, "Risk": 0.08, "Fraud": 0.02},
        "Risk":   {"Normal": 0.20, "Risk": 0.55, "Fraud": 0.25},
        "Fraud":  {"Normal": 0.03, "Risk": 0.12, "Fraud": 0.85},
    },
    "ieee": {
        "Normal": {"Normal": 0.92, "Risk": 0.06, "Fraud": 0.02},
        "Risk":   {"Normal": 0.30, "Risk": 0.50, "Fraud": 0.20},
        "Fraud":  {"Normal": 0.05, "Risk": 0.10, "Fraud": 0.85},
    },
}


class DigitalTwinSimulator:
    """HMM-based trajectory simulator with optional DRL policy."""

    def __init__(self, domain="paysim"):
        self.domain = domain
        self.transitions = DOMAIN_TRANSITIONS.get(domain, DOMAIN_TRANSITIONS["paysim"])

    def forecast_trajectory(self, initial_state="Normal", steps=20,
                            apply_drl=False):
        np.random.seed()
        state = initial_state
        records = []

        for step in range(steps):
            # Risk score by state
            if state == "Normal":
                score = np.random.uniform(0.02, 0.18)
            elif state == "Risk":
                score = np.random.uniform(0.25, 0.65)
            else:
                score = np.random.uniform(0.70, 0.98)

            # DRL intervention — suppress score after detection
            if apply_drl and state == "Fraud" and step > 3:
                score *= 0.35

            records.append({
                "Step": step + 1,
                "Forecast Time": datetime.now() + timedelta(hours=step * 2),
                "Predicted State": state,
                "Fraud Probability Score": round(score, 4),
            })

            # Transition
            probs = list(self.transitions[state].values())
            states = list(self.transitions[state].keys())
            state = np.random.choice(states, p=probs)

        return pd.DataFrame(records)


def run_drl_forecast(domain="paysim", steps=20):
    """Run baseline vs. DRL-intervention dual forecast."""
    sim = DigitalTwinSimulator(domain)
    baseline = sim.forecast_trajectory("Normal", steps, apply_drl=False)
    threat = sim.forecast_trajectory("Risk", steps, apply_drl=True)
    return baseline, threat


def run_mitigation_comparison(domain="paysim", steps=20):
    """Compare no-intervention vs. DRL mitigation from the same threat origin."""
    sim = DigitalTwinSimulator(domain)
    no_action = sim.forecast_trajectory("Fraud", steps, apply_drl=False)
    with_drl = sim.forecast_trajectory("Fraud", steps, apply_drl=True)
    return no_action, with_drl


# Legacy compat
def get_digital_twin_dashboard_data():
    baseline, threat = run_drl_forecast()
    return {
        "baseline_trajectory": baseline.to_dict(orient="records"),
        "threat_trajectory": threat.to_dict(orient="records"),
    }
