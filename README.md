# Vendor Invoice Intelligence System

An AI-driven portal designed to automate and enhance vendor invoice processing. This project leverages machine learning to forecast freight costs accurately and detect risky or abnormal vendor invoices, reducing financial leakage and manual workload.

## Features
- **Freight Cost Prediction:** Predicts freight cost for a vendor invoice using Invoice Dollars to support budgeting, forecasting, and vendor negotiations.
- **Invoice Manual Approval Flag:** Predicts whether a vendor invoice should be flagged for manual approval based on abnormal cost, freight, or delivery patterns using a Random Forest classifier.
- **Interactive UI:** A Streamlit-based web application providing an easy-to-use interface to interact with the models.

## Tech Stack
- **Frontend:** Streamlit
- **Machine Learning:** scikit-learn, pandas, numpy
- **Model Persistence:** Joblib

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/navya-08-code/Vendor-invoice-Intelligence-system.git
   cd Vendor-invoice-Intelligence-system
   ```
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main Streamlit application dashboard.
- `invoice_flagging/`: Code and model for the Invoice Flagging classification model.
- `freight_cost_prediction/`: Models for Freight Cost Prediction.
- `inference/`: Scripts for running inference on the trained models.
