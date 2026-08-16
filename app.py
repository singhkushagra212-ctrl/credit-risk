from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import joblib
import pandas as pd
from fpdf import FPDF
import numpy as np # Import numpy for potential division by zero handling

app = Flask(__name__)
CORS(app)

MODEL_FILENAME = 'C:/Users/dell/Desktop/credit risk/modelartifactes/credit_risk_portal_model.joblib' # Updated path to model
try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    print(f"[ERROR] Model failed to load: {e}")

# Define features for the model, these should match the training data
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
model_cols = numeric_features + categorical_features

LOAN_PRODUCTS = [
    {"bank": "State Bank of India", "type": "SBI Personal Loan", "min_income": 15000, "min_tenure": 12, "max_tenure": 72, "rate": "11.15%"},
    {"bank": "HDFC Bank", "type": "Express Personal Loan", "min_income": 25000, "min_tenure": 12, "max_tenure": 60, "rate": "10.50%"},
    {"bank": "ICICI Bank", "type": "Personal Loan", "min_income": 30000, "min_tenure": 12, "max_tenure": 72, "rate": "10.75%"},
    {"bank": "Bajaj Finserv", "type": "Flexi Personal Loan", "min_income": 22000, "min_tenure": 12, "max_tenure": 96, "rate": "13.00%"}
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract all necessary data points from the request
        person_age = int(data.get('person_age', 30))
        monthly_income = float(data.get('monthly_income', 0))
        monthly_expenses = float(data.get('monthly_expenses', 0))
        existing_emi = float(data.get('existing_emi', 0))
        income_source = data.get('income_source', 'Salaried')
        months_to_pay = int(data.get('months_to_pay', 12))
        person_home_ownership = data.get('person_home_ownership', 'RENT')
        person_emp_length = float(data.get('person_emp_length', 5))
        loan_intent = data.get('loan_intent', 'PERSONAL')
        loan_grade = data.get('loan_grade', 'A')
        loan_amnt = float(data.get('loan_amnt', 0))
        loan_int_rate = float(data.get('loan_int_rate', 10.5))
        cb_person_default_on_file = data.get('cb_person_default_on_file', 'N')
        cb_person_cred_hist_length = int(data.get('cb_person_cred_hist_length', 5))

        # Reconstruct applicant_data as per the notebook's analyze_risk_portal input
        applicant_data = {
            'person_age': person_age,
            'person_income': monthly_income * 12,
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'existing_emi': existing_emi,
            'income_source': income_source,
            'months_to_pay': months_to_pay,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_amnt / (monthly_income * 12) if monthly_income * 12 > 0 else 999.0, # Use a high value for unrealistic ratio
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }

        # --- Replicating analyze_risk_portal logic here --- #
        MIN_INCOME_THRESHOLD = 10000

        # NEW: Minimum Income Threshold Check
        if monthly_income < MIN_INCOME_THRESHOLD:
            decision, risk, suggestion = "Declined", "Critical Risk", f"Monthly income below minimum required threshold of Rs. {MIN_INCOME_THRESHOLD:,.2f}."
        # 1. Unrealistic Age Check
        elif not (18 <= applicant_data['person_age'] <= 80):
            decision, risk, suggestion = "Declined", "Critical Risk", "Applicant age is outside realistic working age range (18-80)."
        # NEW: Age-Based Loan Repayment Limit
        elif (applicant_data['person_age'] + (months_to_pay / 12)) > 70:
            decision, risk, suggestion = "Declined", "Critical Risk", "Applicant's age plus loan tenure exceeds the maximum repayment age limit (70 years)."
        # 2. Unrealistic Employment Length Check (relative to age)
        elif applicant_data['person_emp_length'] < 0 or applicant_data['person_emp_length'] > (applicant_data['person_age'] - 16):
            decision, risk, suggestion = "Declined", "Critical Risk", "Employment length is unrealistic for applicant's age or is negative."
        # 3. Unrealistic Loan-to-Income Ratio
        elif monthly_income <= 0 or applicant_data['loan_percent_income'] > 1.0: # Check the calculated loan_percent_income
            decision, risk, suggestion = "Declined", "Critical Risk", "Loan amount exceeds 100% of annual income or income is zero/negative."
        else:
            input_df = pd.DataFrame([applicant_data])
            # 2. Base ML Probability using the loaded model
            prob = model.predict_proba(input_df[model_cols])[0][1]

            # 3. Real-World Adjustments (Occupation & DTI)
            volatility = {
                'Student': 0.25,
                'Small Business': 0.15,
                'Freelancer': 0.10,
                'Pensioner': 0.05,
                'Others': 0.15
            }.get(income_source, 0)

            prob_adjusted = prob + volatility # Apply volatility to prob

            # Calculate Monthly Payability
            annual_rate = applicant_data['loan_int_rate'] / 100
            # months = applicant_data['months_to_pay'] # Already defined above as months_to_pay
            if months_to_pay == 0:
                months_to_pay = 1 # Default to 1 month to avoid division by zero for EMI calculation

            total_payable = applicant_data['loan_amnt'] * (1 + (annual_rate * (months_to_pay/12)))
            new_emi = total_payable / months_to_pay

            # True Disposable Income accounts for existing debts
            true_disposable = monthly_income - monthly_expenses - existing_emi
            dti_ratio = (new_emi + existing_emi) / monthly_income if monthly_income > 0 else 1.0
            payback_burden = new_emi / true_disposable if true_disposable > 0 else 999.0 # Use a high value for unrealistic burden

            # 4. Stress Test (20% Expense Hike)
            stress_disposable = (monthly_income - (monthly_expenses * 1.20) - existing_emi)
            stress_pass = "PASS" if (new_emi < stress_disposable * 0.7) else "FAIL"

            # 5. Final Decision Logic based on notebook's analyze_risk_portal
            if cb_person_default_on_file == 'Y':
                decision, risk, suggestion = "Review Required", "Medium Risk", "Applicant has a history of default. Additional scrutiny needed."
            elif true_disposable <= 0:
                decision, risk, suggestion = "Declined", "Critical Risk", "Applicant has negative monthly cash flow."
            elif payback_burden > 0.60 or dti_ratio > 0.50:
                decision, risk, suggestion = "Declined", "High Risk", "Total debt obligations exceed safe income thresholds."
            elif prob_adjusted > 0.50:
                decision, risk, suggestion = "Review Required", "Medium Risk", "ML pattern detection suggests high volatility."
            else:
                decision, risk, suggestion = "Approved", "Low Risk", "Strong financial buffer and payback feasibility."

        # Bank suggestions filtering (only if not declined by initial checks)
        filtered_loan_products = []
        if decision not in ["Declined", "Review Required"]:
            filtered_loan_products = [
                p for p in LOAN_PRODUCTS
                if monthly_income >= p['min_income'] and p['min_tenure'] <= months_to_pay <= p['max_tenure']
            ][:3]


        return jsonify({
            "status": decision,
            "risk": risk,
            "details": {
                "emi": round(new_emi, 2) if 'new_emi' in locals() else 0,
                "dti": round(dti_ratio * 100, 2) if 'dti_ratio' in locals() else 0,
                "disposable": round(true_disposable, 2) if 'true_disposable' in locals() else 0,
                "existing_emi": round(existing_emi, 2),
                "stress_pass": stress_pass if 'stress_pass' in locals() else 'N/A'
            },
            "plan": suggestion,
            "bank_suggestions": filtered_loan_products
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.json
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Credit Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Final Decision: {data['status']}", ln=True)
    pdf.cell(200, 10, f"Risk Level: {data['risk']}", ln=True)
    # Access details from the nested structure
    pdf.cell(200, 10, f"Calculated EMI: Rs. {data['details']['emi']}", ln=True)
    pdf.cell(200, 10, f"DTI Ratio: {data['details']['dti']}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Improvement Plan / Professional Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, data['plan'])

    response = Response(pdf.output(dest='S'))
    response.headers.set('Content-Disposition', 'attachment', filename='Loan_Report.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)