# Installation & Setup Guide

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: At least 2GB free
- **Processor**: 4+ cores recommended

## Step-by-Step Installation

### Step 1: Check Python Installation

Open terminal/PowerShell and verify Python version:

```bash
python --version
```

Should show Python 3.8 or higher. If not, download from [python.org](https://www.python.org)

### Step 2: Download Project

Option A: Clone from Git
```bash
git clone <repository-url>
cd FRAUD_DETECTION
```

Option B: Download ZIP
- Extract to desired location
- Navigate to folder in terminal

### Step 3: Create Virtual Environment

Create isolated Python environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in terminal.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 2-5 minutes. Wait for completion.

## Verification

Check if all packages installed correctly:

```bash
python -c "import streamlit; import pandas; import sklearn; print('All packages installed!')"
```

If no errors, you're ready to go!

## Running the Application

### Quick Start

```bash
streamlit run app.py
```

### What to Expect

1. Terminal shows: "You can now view your Streamlit app in your browser"
2. Browser opens automatically to `http://localhost:8501`
3. If not, manually open that URL

### First Time Setup

**Initial Launch:**
1. Let dependencies load (first run takes 30-60 seconds)
2. Dashboard appears with empty state
3. Generate sample UPI dataset to get started

## Getting Started with Dashboard

### Quick Demo (5 minutes)

1. **Load Data**
   - Go to Section 1
   - Click "Load & Preprocess Data" (default 5000 records)
   - To support large CSV uploads (up to 500 MB) launch Streamlit with:
     ```bash
     streamlit run app.py --server.maxUploadSize=500
     ```
     or set `maxUploadSize` under `[server]` in `~/.streamlit/config.toml`.
   - Wait for success message

2. **Train Models**
   - Go to Section 2
   - Click "Train All Models"
   - Wait 2-3 minutes for training

3. **Make Prediction**
   - Go to Section 3
   - Click "Sample Transaction" or enter custom data
   - Click "Predict Fraud"
   - See results instantly

## Troubleshooting

### Issue: "Command not found: streamlit"

**Windows:**
```bash
python -m streamlit run app.py
```

**macOS/Linux:**
```bash
python3 -m streamlit run app.py
```

### Issue: "Port 8501 already in use"

Use different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Module not found" after install

Ensure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Issue: ImportError for tensorflow

TensorFlow is optional. If error appears:
```bash
pip install --upgrade tensorflow
```

Or use models without Autoencoder.

### Issue: Out of Memory

In `config.py`, change:
```python
"sample_size": 50000  # Change to 10000
```

## Managing Virtual Environment

### Deactivate Environment
```bash
deactivate
```

### Reactivate Later
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Export Requirements
If you modify packages:
```bash
pip freeze > requirements.txt
```

## Deployment Options

### Option 1: Local Machine (Current)
- Run on personal computer
- Accessible at `http://localhost:8501`

### Option 2: Streamlit Cloud (Free)
```bash
# Push repo to GitHub
# Go to share.streamlit.io
# Connect repository
# Deploy
```

### Option 3: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

### Option 4: Heroku

```bash
# Install Heroku CLI
# Login: heroku login
# Create app: heroku create my-fraud-app
# Deploy: git push heroku main
```

## Using Different Datasets

### Option 1: PaySim Dataset (Kaggle)

Requires Kaggle API setup:
```bash
pip install kagglehub

# Set API credentials:
# Create ~/.kaggle/kaggle.json with credentials
```

Then in dashboard, upload existing paysimdataset.

### Option 2: Custom CSV

Format your CSV with required columns:
- A numeric column for amounts
- A categorical column for transaction type
- A time/date column
- A fraud label column (named: fraud, isFraud, or Class)

Upload in Section 1.

### Option 3: Generate Synthetics

Dashboard generates realistic UPI datasets automatically:
- 5000 transactions by default
- 2% fraud rate
- Realistic patterns

## Learning Resources

### Understanding the Code
1. Start with `app.py` - main dashboard logic
2. Check `modules/` - individual components
3. Read function docstrings - embedded documentation
4. See `config.py` - all settings in one place

### Python Learning
- [Python Docs](https://docs.python.org/3/)
- [Streamlit Tutorial](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)

### Machine Learning
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Models](https://scikit-learn.org/stable/modules/classes.html)

## Configuration Customization

All settings in `config.py`:

```python
# Change default dataset size
UPI_CONFIG = {
    "num_records": 10000,  # Increase for more data
    "fraud_rate": 0.05,    # 5% fraud instead of 2%
}

# Tune models
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,  # More trees
        "max_depth": 20,      # Deeper trees
    }
}
```

## Security Notes

### For Production Use:
1. Use environment variables for sensitive data
2. Implement user authentication
3. Add database layer for persistence
4. Use HTTPS for network communication
5. Implement logging and monitoring
6. Add data encryption

### Current Setup:
 - Local-only processing
 - No external data transmission
 - Synthetic datasets for safety
 - Passwords/tokens visible in code (update for production)

## Performance Tips

### Speed Up Training:
1. Use fewer records (reduce `sample_size`)
2. Use Random Forest instead of XGBoost
3. Skip Autoencoder (reduce models to 3)
4. Disable SHAP explanations

### Speed Up Predictions:
1. Use Logistic Regression model
2. Pre-process batch data
3. Cache results

### Free Up Memory:
1. Close other applications
2. Clear browser cache
3. Restart Python kernel

## Next Steps

1. **Explore:** Try different datasets
2. **Customize:** Modify for your use case
3. **Integrate:** Connect to real data source
4. **Deploy:** Use Streamlit Cloud or Docker
5. **Monitor:** Track model performance
6. **Improve:** Retrain with new data

## Getting Help

### Common Questions:

**Q: How do I stop the app?**
A: Press `Ctrl+C` in terminal

**Q: Can I run multiple instances?**
A: Yes, use different ports: `--server.port 8502`

**Q: How do I save my models permanently?**
A: Models auto-save in `models/` folder

**Q: Can I use my existing ML model?**
A: Yes, save as .pkl and load with `load_latest_model()`

**Q: How often should I retrain?**
A: Weekly (with new fraud patterns) or monthly (routine update)

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| ModuleNotFoundError | Package missing | `pip install -r requirements.txt` |
| FileNotFoundError | Path incorrect | Check working directory |
| MemoryError | Too much data | Reduce sample_size |
| TensorflowError | GPU/CPU issue | Install CPU version |
| PortInUseError | Port occupied | Use `--server.port 8502` |

## Monitoring Performance

Track in dashboard:
- Model accuracy metrics
- Training time
- Prediction latency
- Memory usage

Monitor in production:
- False positive rate
- False negative rate
- User feedback
- Fraud evasion patterns

---

**Ready to start?** Run: `streamlit run app.py`

Questions? Check README.md or code comments!

Enjoy building your fraud detection system!
