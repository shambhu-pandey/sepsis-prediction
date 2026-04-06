import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

def generate_banksim(num_rows=25000):
    print("Synthesizing BankSim Retail Data...")
    np.random.seed(42)
    categories = ["es_transportation", "es_health", "es_otherservices", "es_food", 
                  "es_hotelservices", "es_tech", "es_sportsandtoys", "es_wellnessandbeauty", 
                  "es_hyper", "es_fashion", "es_barsandrestaurants", "es_travel", "es_leisure"]
                  
    data = {
        "step": np.random.randint(1, 180, num_rows),
        "customer": [f"'C{np.random.randint(10000, 99999)}'" for _ in range(num_rows)],
        "age": np.random.choice(["0", "1", "2", "3", "4", "5", "6", "U"], num_rows),
        "gender": np.random.choice(["F", "M", "E", "U"], num_rows, p=[0.45, 0.45, 0.05, 0.05]),
        "merchant": [f"'M{np.random.randint(10000, 99999)}'" for _ in range(num_rows)],
        "category": np.random.choice(categories, num_rows),
        "amount": np.random.exponential(scale=35.0, size=num_rows).round(2),
        "fraud": [0] * num_rows  # Default honest
    }
    
    df = pd.DataFrame(data)
    
    # Inject fraud patterns (e.g. extreme amounts in es_travel, es_tech)
    fraud_indices = np.random.choice(df.index, size=int(num_rows * 0.03), replace=False)
    df.loc[fraud_indices, "fraud"] = 1
    df.loc[fraud_indices, "amount"] = np.random.uniform(500, 5000, len(fraud_indices)).round(2)
    df.loc[fraud_indices, "category"] = np.random.choice(["es_travel", "es_tech", "es_sportsandtoys"], len(fraud_indices))
    
    df.to_csv("data/banksim.csv", index=False)
    print("BankSim simulation complete -> data/banksim.csv")

def generate_ieee(num_rows=25000):
    print("Synthesizing IEEE-CIS E-Commerce Data...")
    np.random.seed(42)
    
    data = {
        "TransactionDT": np.random.randint(86400, 15811131, num_rows),
        "TransactionAmt": np.random.exponential(scale=135.0, size=num_rows).round(2),
        "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], num_rows),
        "card1": np.random.randint(1000, 18396, num_rows),
        "card2": np.random.randint(100, 600, num_rows),
        "card3": np.random.randint(100, 200, num_rows),
        "card4": np.random.choice(["visa", "mastercard", "american express", "discover"], num_rows),
        "card5": np.random.randint(100, 226, num_rows),
        "card6": np.random.choice(["credit", "debit"], num_rows),
        "addr1": np.random.randint(100, 500, num_rows),
        "addr2": np.random.randint(10, 100, num_rows),
        "dist1": np.random.exponential(scale=50.0, size=num_rows).round(1),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"], num_rows),
        "DeviceType": np.random.choice(["desktop", "mobile"], num_rows),
        "DeviceInfo": ["Windows" if np.random.random() > 0.5 else "iOS" for _ in range(num_rows)],
        "isFraud": [0] * num_rows
    }
    
    df = pd.DataFrame(data)
    
    # Inject fraud patterns (e.g. C or H type products, specific domains, high amounts)
    fraud_indices = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
    df.loc[fraud_indices, "isFraud"] = 1
    df.loc[fraud_indices, "ProductCD"] = np.random.choice(["C", "H", "S"], len(fraud_indices))
    df.loc[fraud_indices, "TransactionAmt"] = np.random.uniform(500, 10000, len(fraud_indices)).round(2)
    df.loc[fraud_indices, "P_emaildomain"] = "anonymous.com"
    df.loc[fraud_indices, "DeviceType"] = "mobile"
    df.loc[fraud_indices, "dist1"] = np.random.uniform(500, 2000, len(fraud_indices)).round(1)

    df.to_csv("data/ieee.csv", index=False)
    print("IEEE-CIS simulation complete -> data/ieee.csv")

if __name__ == "__main__":
    generate_banksim()
    generate_ieee()
