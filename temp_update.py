import re

with open("modules/fraud_explainer.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Strip the duplicate headers from display_fraud_explanation
pattern_headers = r'    # Header based on prediction\n    if explanation\["is_fraud"\]:.*?st\.markdown\("---"\)\n'
code = re.sub(pattern_headers, '', code, flags=re.DOTALL)

# 2. Add the routing for banksim and ieee
routing_old = """    # Analyze based on dataset type
    if dataset_type == "paysim":
        explanation = _explain_paysim_fraud(transaction_data, feature_values, prediction, probability)
    else:
        explanation = _explain_cicids_fraud(transaction_data, feature_values, prediction, probability)"""

routing_new = """    # Analyze based on dataset type
    if dataset_type == "paysim":
        explanation = _explain_paysim_fraud(transaction_data, feature_values, prediction, probability)
    elif dataset_type == "banksim":
        explanation = _explain_banksim_fraud(transaction_data, feature_values, prediction, probability)
    elif dataset_type == "ieee":
        explanation = _explain_ieee_fraud(transaction_data, feature_values, prediction, probability)
    else:
        explanation = _explain_cicids_fraud(transaction_data, feature_values, prediction, probability)"""

code = code.replace(routing_old, routing_new)

# 3. Inject new _explain_banksim_fraud and _explain_ieee_fraud before display_fraud_explanation
banksim_ieee_code = """
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
            "issue": "Extremely high purchase amount for retail categorization",
            "tip": "Verify this large retail purchase with your bank."
        })
    elif amount > 300:
        risk_factors.append({
            "factor": "Transaction Amount", "value": f"${amount:,.2f}",
            "issue": "Higher than normal retail spending behavior",
            "tip": "Ensure you recognize this charge."
        })
        
    if category in ["es_travel", "es_hotelservices", "es_leisure"]:
        risk_factors.append({
            "factor": "Merchant Category", "value": category.replace('es_','').title(),
            "issue": "High-risk category (travel/leisure often targeted by fraudsters)",
            "tip": "Confirm travel reservations align with your itinerary."
        })
        
    explanation["main_reasons"] = risk_factors
    if risk_factors:
        reasons_text = [f"• {rf['factor']}: {rf['issue']}" for rf in risk_factors]
        explanation["simple_explanation"] = f"This BankSim transaction shows unusual patterns:\\n\\n" + "\\n".join(reasons_text)
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
            "issue": "Very high online card-not-present transaction amount",
            "tip": "Verify large e-commerce orders."
        })
    risk_factors.append({
        "factor": "Device/Browser Signature", "value": "Anonymous",
        "issue": "V1-V339 features and device identity checks performed",
        "tip": "Ensure purchases were made from trusted devices."
    })
        
    explanation["main_reasons"] = risk_factors
    reasons_text = [f"• {rf['factor']}: {rf['issue']}" for rf in risk_factors]
    explanation["simple_explanation"] = f"This IEEE-CIS e-commerce payment yielded these insights:\\n\\n" + "\\n".join(reasons_text)
    explanation["recommendations"] = ["Enable 2FA for online purchases", "Use virtual credit cards"]
    return explanation

def display_fraud_explanation(explanation, dataset_type="paysim"):
"""

code = code.replace("def display_fraud_explanation(explanation, dataset_type=\"paysim\"):", banksim_ieee_code)

with open("modules/fraud_explainer.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Updated successfully")
