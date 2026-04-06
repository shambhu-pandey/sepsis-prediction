"""
Fraud Explanation Module - Simple explanations for non-technical users
"""

import numpy as np
import pandas as pd
import streamlit as st


def get_simple_fraud_explanation(transaction_data, feature_values, prediction, probability, dataset_type="paysim"):
    """
    Generate simple, easy-to-understand fraud explanation for non-technical users.
    
    Args:
        transaction_data: Original transaction details
        feature_values: Feature values used for prediction
        prediction: 0 = legitimate, 1 = fraud
        probability: Fraud probability (0-1)
        dataset_type: "paysim" or "cicids"
    
    Returns:
        dict: Simple explanation with reasons and recommendations
    """
    
    explanation = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "main_reasons": [],
        "what_went_wrong": [],
        "simple_explanation": "",
        "recommendations": [],
        "technical_details": {}
    }
    
    # Analyze based on dataset type
    if dataset_type == "paysim":
        explanation = _explain_paysim_fraud(transaction_data, feature_values, prediction, probability)
    elif dataset_type == "banksim":
        explanation = _explain_banksim_fraud(transaction_data, feature_values, prediction, probability)
    elif dataset_type == "ieee":
        explanation = _explain_ieee_fraud(transaction_data, feature_values, prediction, probability)
    else:
        explanation = _explain_cicids_fraud(transaction_data, feature_values, prediction, probability)
    
    return explanation


def _explain_paysim_fraud(transaction_data, feature_values, prediction, probability):
    """Explain PaySim/Payment fraud in simple terms."""
    
    explanation = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "main_reasons": [],
        "what_went_wrong": [],
        "simple_explanation": "",
        "recommendations": [],
        "technical_details": {}
    }
    
    # Extract values with defaults
    amount = transaction_data.get("amount", 0)
    hour = transaction_data.get("hour", 12)
    device_age = transaction_data.get("device_age_days", 365)
    txn_count = transaction_data.get("transaction_count_24h", 1)
    location_change = transaction_data.get("location_change_indicator", 0)
    tx_type = transaction_data.get("transaction_type", "P2P")
    sender_device = transaction_data.get("sender_device_type", "Android")
    
    # Analyze each factor
    risk_factors = []
    recommendations = []
    
    # Check Amount - with special context for large amounts (tuition, property, etc.)
    if amount > 500000:  # Very large (5+ lakhs)
        risk_factors.append({
            "factor": "Transaction Amount",
            "value": f"₹{amount:,.2f} (₹{amount/100000:.1f} Lakhs)",
            "issue": "Very high value transaction - requires additional verification",
            "tip": "This is normal for big payments like property, education (tuition), or business",
            "context": "Banks flag large amounts as a safety measure, even for legitimate payments"
        })
        recommendations.append("For large payments like tuition fees (₹5-20 lakhs), this is expected behavior")
        recommendations.append("You can complete this by verifying with your bank")
        recommendations.append("Call your bank's customer care to pre-approve large transfers")
    elif amount > 100000:  # Large (1-5 lakhs)
        risk_factors.append({
            "factor": "Transaction Amount",
            "value": f"₹{amount:,.2f} (₹{amount/100000:.1f} Lakhs)",
            "issue": "Large transaction amount - common for education fees, purchases",
            "tip": "For payments like tuition fees, this is common - just verify your identity"
        })
        recommendations.append("Large education payments (₹1-10 lakhs) are common and safe when verified")
        recommendations.append("Complete with OTP verification")
    elif amount > 50000:
        risk_factors.append({
            "factor": "Transaction Amount",
            "value": f"₹{amount:,.2f}",
            "issue": "Very high amount - much higher than usual",
            "tip": "For large transactions, verify with OTP or phone call"
        })
        recommendations.append("Enable transaction alerts for amounts above ₹10,000")
        recommendations.append("Use dual verification for amounts above ₹50,000")
    elif amount > 20000:
        risk_factors.append({
            "factor": "Transaction Amount",
            "value": f"₹{amount:,.2f}",
            "issue": "Above average amount",
            "tip": "Consider splitting large transactions"
        })
    
    # Check Device Age
    if device_age < 7:
        risk_factors.append({
            "factor": "Device Age",
            "value": f"{device_age:.0f} days",
            "issue": "New device - account may be compromised",
            "tip": "Register new devices through banking app"
        })
        recommendations.append("Never share OTP with anyone")
        recommendations.append("Enable device authentication")
    elif device_age < 30:
        risk_factors.append({
            "factor": "Device Age",
            "value": f"{device_age:.0f} days",
            "issue": "Relatively new device",
            "tip": "Ensure you recognize this device"
        })
    
    # Check Transaction Time
    if hour < 6 or hour > 22:
        risk_factors.append({
            "factor": "Transaction Time",
            "value": f"{hour}:00",
            "issue": "Unusual time - most fraud happens at night",
            "tip": "Set transaction limits for night hours"
        })
        recommendations.append("Enable 'Do Not Disturb' hours in your banking app")
    
    # Check Transaction Count (Velocity)
    if txn_count > 10:
        risk_factors.append({
            "factor": "Transaction Frequency",
            "value": f"{txn_count} transactions in 24 hours",
            "issue": "Too many transactions - possible account takeover",
            "tip": "Freeze account if you didn't make these transactions"
        })
        recommendations.append("Review your transaction history immediately")
    elif txn_count > 5:
        risk_factors.append({
            "factor": "Transaction Frequency",
            "value": f"{txn_count} transactions in 24 hours",
            "issue": "Higher than normal activity",
            "tip": "Monitor your account closely"
        })
    
    # Check Location Change
    if location_change == 1:
        risk_factors.append({
            "factor": "Location",
            "value": "Different location detected",
            "issue": "Transaction from new/unusual location",
            "tip": "Verify if you're traveling to a new place"
        })
        recommendations.append("Enable location-based security")
    
    # Build explanation
    explanation["main_reasons"] = risk_factors
    
    if risk_factors:
        # Create simple explanation
        reasons_text = []
        for rf in risk_factors:
            reasons_text.append(f"• {rf['factor']}: {rf['issue']}")
        
        explanation["simple_explanation"] = (
            f"This transaction has been flagged as {'HIGH RISK' if probability > 0.7 else 'SUSPICIOUS'} "
            f"because {len(risk_factors)} factor(s) {'are' if len(risk_factors) > 1 else 'is'} unusual:\n\n"
            + "\n".join(reasons_text)
        )
        explanation["what_went_wrong"] = [rf["issue"] for rf in risk_factors]
    else:
        explanation["simple_explanation"] = (
            "This transaction appears normal based on typical patterns. "
            "However, always stay vigilant!"
        )
    
    # Add recommendations
    if not recommendations:
        recommendations = [
            "Continue monitoring your transactions regularly",
            "Keep your app updated for latest security features"
        ]
    explanation["recommendations"] = recommendations
    
    # Technical details
    explanation["technical_details"] = {
        "amount": amount,
        "hour": hour,
        "device_age_days": device_age,
        "transaction_count_24h": txn_count,
        "location_change": bool(location_change),
        "transaction_type": tx_type,
        "sender_device": sender_device
    }
    
    return explanation


def _explain_cicids_fraud(transaction_data, feature_values, prediction, probability):
    """Explain CICIDS/Network attack in simple terms."""
    
    explanation = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "main_reasons": [],
        "what_went_wrong": [],
        "simple_explanation": "",
        "recommendations": [],
        "technical_details": {}
    }
    
    # Network traffic analysis
    risk_factors = []
    recommendations = []
    
    # Common network attack indicators
    if "packet_count" in feature_values:
        packet_count = feature_values.get("packet_count", 0)
        if packet_count > 1000:
            risk_factors.append({
                "factor": "High Packet Volume",
                "value": f"{packet_count} packets",
                "issue": "Extremely high data volume - resembles a traffic overload attack",
                "tip": "Check if this device is downloading a massive update; otherwise it could be infected."
            })
            recommendations.append("Implement traffic rate limiting")
            recommendations.append("Enable network overload protection")
    
    if "flow_duration" in feature_values:
        duration = feature_values.get("flow_duration", 0)
        if duration > 100000:
            risk_factors.append({
                "factor": "Connection Duration",
                "value": f"{duration/1000:.1f} seconds",
                "issue": "Unusually long connection running silently in the background",
                "tip": "Ensure you recognize what this device is persistently connecting to."
            })
    
    if "byte_count" in feature_values:
        bytes_sent = feature_values.get("byte_count", 0)
        if bytes_sent > 1000000:
            risk_factors.append({
                "factor": "Data Transfer",
                "value": f"{bytes_sent/1024/1024:.1f} MB",
                "issue": "A huge amount of data was sent out - possible file theft",
                "tip": "Verify if someone legitimately uploaded large files."
            })
            recommendations.append("Monitor all outgoing file uploads")
            recommendations.append("Implement file theft prevention rules")
    
    # Protocol analysis
    if "protocol" in feature_values:
        protocol = feature_values.get("protocol", "TCP")
        if protocol in ["UDP"]:
            risk_factors.append({
                "factor": "Protocol",
                "value": protocol,
                "issue": "This connection method is less secure and often favored by hackers",
                "tip": "Ensure this program is supposed to be running freely."
            })
    
    # Port analysis
    if "port" in feature_values:
        port = feature_values.get("port", 0)
        suspicious_ports = [23, 135, 139, 445, 1433, 3389]  # Telnet, RPC, SMB, SQL, RDP
        if port in suspicious_ports:
            port_names = {23: "Telnet", 135: "RPC", 139: "SMB", 445: "SMB", 1433: "SQL Server", 3389: "RDP"}
            risk_factors.append({
                "factor": "Entry Point (Port) Used",
                "value": f"{port} ({port_names.get(port, 'Common')})",
                "issue": "This digital entry point is commonly scanned and broken into by hackers",
                "tip": "Close unused computer connections and features"
            })
            recommendations.append(f"Review necessity of {port_names.get(port, 'port ' + str(port))}")
            recommendations.append("Use firewall to restrict access")
    
    # Build explanation
    explanation["main_reasons"] = risk_factors
    
    if risk_factors:
        reasons_text = []
        for rf in risk_factors:
            reasons_text.append(f"• {rf['factor']}: {rf['issue']}")
        
        explanation["simple_explanation"] = (
            f"This network traffic has been flagged as {'HIGH RISK' if probability > 0.7 else 'SUSPICIOUS'} "
            f"because:\n\n" + "\n".join(reasons_text)
        )
        explanation["what_went_wrong"] = [rf["issue"] for rf in risk_factors]
    else:
        explanation["simple_explanation"] = (
            "This network traffic appears normal based on typical patterns."
        )
    
    if not recommendations:
        recommendations = [
            "Continue monitoring network traffic",
            "Keep security systems updated",
            "Review logs regularly"
        ]
    explanation["recommendations"] = recommendations
    
    explanation["technical_details"] = feature_values
    
    return explanation



def _explain_banksim_fraud(transaction_data, feature_values, prediction, probability):
    explanation = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "main_reasons": [], "what_went_wrong": [], "simple_explanation": "", "recommendations": [], "technical_details": {}
    }
    amount = transaction_data.get("amount", 0)
    category = transaction_data.get("category", "")
    risk_factors = []
    
    if amount > 1000:
        risk_factors.append({
            "factor": "Transaction Amount", "value": f"${amount:,.2f}",
            "issue": "This is a very large amount of money for a typical shopping trip",
            "tip": "Verify this purchase if you didn't buy anything expensive recently."
        })
    elif amount > 300:
        risk_factors.append({
            "factor": "Transaction Amount", "value": f"${amount:,.2f}",
            "issue": "Higher spending than your normal shopping routine",
            "tip": "Ensure you recognize this store and amount."
        })
        
    if category in ["es_travel", "es_hotelservices", "es_leisure"]:
        risk_factors.append({
            "factor": "Merchant Category", "value": category.replace('es_','').title(),
            "issue": "This merchant type (like travel bookings) is a favorite target for scammers",
            "tip": "Double-check that you actually booked this."
        })
        
    explanation["main_reasons"] = risk_factors
    if risk_factors:
        reasons_text = [f"• {rf['factor']}: {rf['issue']}" for rf in risk_factors]
        explanation["simple_explanation"] = f"This BankSim transaction shows unusual patterns:\n\n" + "\n".join(reasons_text)
    else:
        explanation["simple_explanation"] = "This retail transaction aligns with typical legitimate spending rules."
    explanation["recommendations"] = ["Review monthly credit statements", "Set spending limits"]
    return explanation

def _explain_ieee_fraud(transaction_data, feature_values, prediction, probability):
    explanation = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "main_reasons": [], "what_went_wrong": [], "simple_explanation": "", "recommendations": [], "technical_details": {}
    }
    amount = transaction_data.get("TransactionAmt", 0)
    card_type = transaction_data.get("card4", "Credit")
    risk_factors = []
    
    if amount > 500:
        risk_factors.append({
            "factor": "Online Purchase Amount", "value": f"${amount:,.2f}",
            "issue": "Very high cost for an online or internet purchase",
            "tip": "Check your purchase history to verify this order."
        })
    risk_factors.append({
        "factor": "Unfamiliar Checkout", "value": "Anonymous",
        "issue": "This purchase was made from an unfamiliar device or location",
        "tip": "Ensure nobody else has access to your online shopping accounts."
    })
        
    explanation["main_reasons"] = risk_factors
    reasons_text = [f"• {rf['factor']}: {rf['issue']}" for rf in risk_factors]
    explanation["simple_explanation"] = f"This online shopping payment showed these warning signs:\n\n" + "\n".join(reasons_text)
    explanation["recommendations"] = ["Enable 2FA for online purchases", "Use virtual credit cards"]
    return explanation

def display_fraud_explanation(explanation, dataset_type="paysim"):

    """
    Display fraud explanation in a user-friendly way using Streamlit.
    
    Args:
        explanation: The explanation dict from get_simple_fraud_explanation
        dataset_type: "paysim" or "cicids"
    """
    
    # Simple Explanation
    st.markdown("### What Happened?")
    st.markdown(explanation["simple_explanation"])
    
    # Detailed Reasons
    if explanation["main_reasons"]:
        st.markdown("### Why Was This Flagged?")
        
        for i, reason in enumerate(explanation["main_reasons"], 1):
            with st.expander(f"**{i}. {reason['factor']}**", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Value:** {reason['value']}")
                with col2:
                    st.markdown(f"**Issue:** {reason['issue']}")
                st.info(f"{reason['tip']}")
                # Show additional context for large amounts (tuition, property, etc.)
                if "context" in reason:
                    st.success(f"{reason['context']}")
    
    # Recommendations
    if explanation["recommendations"]:
        st.markdown("### How to Stay Safe")
        for rec in explanation["recommendations"]:
            # remove any leading decorative emoji from recommendation strings
            clean_rec = rec
            for ch in []:
                clean_rec = clean_rec.replace(ch, "")
            st.markdown(f"- {clean_rec}")
    
    # Technical Details (hidden for privacy)
    # Do not display raw technical fields from pre-trained samples.
    st.info("Technical details hidden for privacy.")


def get_prevention_tips(dataset_type="paysim"):
    """
    Get general fraud prevention tips.
    
    Args:
        dataset_type: "paysim" or "cicids"
    
    Returns:
        list: Prevention tips
    """
    if dataset_type == "paysim":
        return [
            "Enable two-factor authentication (2FA) for all transactions",
            "Set up transaction alerts for amounts above ₹1,000",
            "Keep your banking app updated to the latest version",
            "Never share OTP, PIN, or password with anyone",
            "Be cautious of calls asking for bank details",
            "Verify recipient before making payments",
            "Use secure internet connection (avoid public Wi-Fi for banking)",
            "Review your transaction history regularly"
        ]
    else:
        return [
            "Use a robust firewall to filter traffic",
            "Implement encryption for all sensitive data",
            "Monitor network traffic for unusual patterns",
            "Use strong passwords and change them regularly",
            "Close unused ports and disable unnecessary services",
            "Keep all software and systems updated",
            "Follow the principle of least privilege for access",
            "Conduct regular security audits"
        ]
