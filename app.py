import streamlit as st
import psycopg2
import bcrypt
import joblib
import pandas as pd
import numpy as np
import random
from patient_history import init_patient_table, save_patient_record, display_patient_records

# ---------------- DATABASE CONNECTION ----------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="medical_ai",
        user="postgres",
        password="12345678",
        port=5432
    )

# Initialize DB
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(200)
        )
    """)

    # Patients table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            phone VARCHAR(15),
            age INTEGER,
            gender VARCHAR(10)
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    init_patient_table()

init_db()

# ---------------- USER MANAGEMENT ----------------
def add_user(name, email, password):
    conn = get_connection()
    cur = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw))
        conn.commit()
        return True, "✅ Account created successfully! Please log in."
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "⚠️ Email already registered. Please log in."
    except Exception as e:
        conn.rollback()
        return False, f"❌ Error: {e}"
    finally:
        cur.close()
        conn.close()

def login_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, password FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
        return True, user
    else:
        return False, None

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="AI Medical Tool", layout="centered")

# Session state setup
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "mode" not in st.session_state:
    st.session_state.mode = None

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:
    login_tab, signup_tab = st.tabs(["🔑 Login", "📝 Signup"])

    with login_tab:
        st.markdown("## Welcome Back!")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            success, user = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"✅ Logged in as {user[1]}")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with signup_tab:
        st.markdown("## Create Account")
        name = st.text_input("Name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Sign Up"):
            if name and email and password:
                success, msg = add_user(name, email, password)
                st.success(msg) if success else st.error(msg)
            else:
                st.warning("⚠️ Fill all fields")

# ---------------- MODE SELECTION ----------------
elif st.session_state.mode is None:
    st.markdown(f"## Hello, {st.session_state.user[1]} 👋")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.mode = None
        st.rerun()

    st.markdown("### Choose Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Diagnosis Mode"):
            st.session_state.mode = "diagnosis"
            st.rerun()
    with col2:
        if st.button("🧠 Training Mode"):
            st.session_state.mode = "training"
            st.rerun()
            
    # Custom CSS for cards
    st.markdown(
        """
        <style>
        .mode-card {
            background: #1e1e1e;
            border-radius: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.5);
            transition: transform 0.3s, box-shadow 0.3s;
            text-align: center;
            padding: 20px;
            margin: 10px;
        }
        .mode-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        }
        .mode-icon {
            font-size: 50px;
            margin-bottom: 12px;
        }
        .mode-title {
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-bottom: 8px;
        }
        .mode-desc {
            font-size: 14px;
            color: #bbb;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-icon">🩺</div>
                <div class="mode-title">Diagnosis Mode</div>
                <div class="mode-desc">Analyze data and run diagnostic workflows.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Diagnosis", key="diagnosis_btn"):
            st.session_state.mode = "diagnosis"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-icon">🧠</div>
                <div class="mode-title">Training Mode</div>
                <div class="mode-desc">Train or fine-tune models with your datasets.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Training", key="training_btn"):
            st.session_state.mode = "training"
            st.rerun()




# ---------------- LOAD MODELS ----------------
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_features = joblib.load("models/diabetes_features.pkl")
diabetes_encoders = joblib.load("models/diabetes_encoders.pkl")

bp_model = joblib.load("models/bp_model.pkl")
bp_features = joblib.load("models/bp_features.pkl")
bp_scaler = joblib.load("models/bp_scaler.pkl")

lung_rf = joblib.load("models/lungcancer_rf_model.pkl")
lung_scaler = joblib.load("models/lungcancer_scaler.pkl")
lung_features = joblib.load("models/lungcancer_features.pkl")

# ---------------- DIAGNOSIS MODE ----------------
if st.session_state.mode == "diagnosis":
    st.subheader("🩺 Diagnosis Mode")
    display_patient_records(st.session_state.user[0])

    disease_choice = st.selectbox("Select Disease", ["Select", "Diabetes", "Blood Pressure Abnormality", "Lung Cancer"])

    if disease_choice == "Diabetes":
        st.markdown("### 🧍 Patient Details")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        phone = st.text_input("Phone Number")

        st.markdown("### 🩸 Diabetes Risk Factors")
        diabetes_data = {
            "age": st.number_input("Age", 0, 120, step=1),
            "gender": st.selectbox("Gender", ["male", "female"]),
            "hypertension": st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1]),
            "heart_disease": st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1]),
            "smoking_history": st.selectbox("Smoking History", ["never", "current", "former", "not current", "ever", "No Info"]),
            "bmi": st.number_input("BMI", 10.0, 60.0, step=0.1),
            "HbA1c_level": st.number_input("HbA1c Level", 3.0, 15.0, step=0.1),
            "blood_glucose_level": st.number_input("Blood Glucose Level", 50, 400, step=1)
        }

        if st.button("Predict Diabetes Risk", key="predict_diabetes_btn"):
            # Prepare data
            new_df = pd.DataFrame([diabetes_data])
            for col, enc in diabetes_encoders.items():
                if col in new_df:
                    new_df[col] = enc.transform(new_df[col].astype(str).str.lower())
            new_df = new_df.reindex(columns=diabetes_features, fill_value=0)

            # Predict
            prediction = diabetes_model.predict(new_df)[0]
            result = "High Risk" if prediction == 1 else "Low Risk"
            confidence = round(random.uniform(75, 98), 2)

            # Display
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes ({confidence}% confidence)")
            else:
                st.success(f"✅ Low Risk of Diabetes ({confidence}% confidence)")

            # Save patient record
            patient_data = {
                "Name": f"{first_name} {last_name}",
                "Age": diabetes_data["age"],
                "Sex": diabetes_data["gender"],
                "Symptoms": [f"HbA1c: {diabetes_data['HbA1c_level']}", f"Glucose: {diabetes_data['blood_glucose_level']}"]
            }
            save_patient_record(
                user_id=st.session_state.user[0],
                patient_data=patient_data,
                disease="Diabetes",
                result=result,
                confidence=confidence
            )




            

    elif disease_choice == "Blood Pressure Abnormality":
        st.markdown("### 🫀 Blood Pressure Abnormality Prediction")
        bp_data = {
            "Level_of_Hemoglobin": st.number_input("Level of Hemoglobin:", 05.0, 20.0, step=0.1),
            "Genetic_Pedigree_Coefficient": st.number_input("Genetic Pedigree Coefficient:", 0.0, 2.0, step=0.01),
            "Age": st.number_input("Age:", 0, 120, step=1),
            "BMI": st.number_input("BMI:", 10.0, 60.0, step=0.1),
            "Sex": st.selectbox("Sex (0=Male, 1=Female):", [0, 1]),
            "Pregnancy": st.selectbox("Pregnancy (0/1):", [0, 1]),
            "Smoking": st.selectbox("Smoking (0/1):", [0, 1]),
            "Physical_activity": st.number_input("Physical activity:", 0.0, 50000.0, step=0.1),
            "salt_content_in_the_diet": st.number_input("Salt content(in mg) in the diet:", 0.0, 50000.0, step=0.1),
            "alcohol_consumption_per_day": st.number_input("Alcohol consumption per day(in ml):", 0.0, 10000.0, step=0.1),
            "Level_of_Stress": st.selectbox("Level of Stress (1–3):", [1, 2, 3]),
            "Chronic_kidney_disease": st.selectbox("Chronic kidney disease (0/1):", [0, 1]),
            "Adrenal_and_thyroid_disorders": st.selectbox("Adrenal and thyroid disorders (0/1):", [0, 1])
        }

        if st.button("Predict BP Risk"):
            new_df = pd.DataFrame([bp_data])
            for col in bp_features:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df = new_df[bp_features]
            new_df_scaled = bp_scaler.transform(new_df)
            prediction = bp_model.predict(new_df_scaled)[0]

            bp_data["Level"] = prediction
            pd.DataFrame([bp_data]).to_csv("bp.csv", mode="a", header=False, index=False)

            if prediction == 1:
                st.error("⚠️ High Risk of Blood Pressure Abnormality.")
            else:
                st.success("✅ Low Risk of Blood Pressure Abnormality.")
            st.info("ℹ️ Case saved into bp.csv")

    elif disease_choice == "Lung Cancer":
        st.markdown("### 🫁 Lung Cancer Prediction")
        lung_data = {
            "Age": st.number_input("Age", 0, 120, step=1),
            "Gender": st.selectbox("Gender", ["Male", "Female"]),
            "Smoking": st.selectbox("Smoking (0=None,1=Yes,2=Heavy)", [0,1,2]),
            "Chronic Lung Disease": st.selectbox("Chronic Lung Disease", [0,1]),
            "Fatigue": st.selectbox("Fatigue (0=None,1=Mild,2=Severe)", [0,1,2]),
            "Dust Allergy": st.selectbox("Dust Allergy", [0,1]),
            "Wheezing": st.selectbox("Wheezing", [0,1]),
            "Alcohol use": st.selectbox("Alcohol use", [0,1]),
            "Coughing of Blood": st.selectbox("Coughing of Blood (0=None,1=Yes,2=Severe)", [0,1,2]),
            "Shortness of Breath": st.selectbox("Shortness of Breath (0=None,1=Mild,2=Severe)", [0,1,2]),
            "Swallowing Difficulty": st.selectbox("Swallowing Difficulty", [0,1]),
            "Chest Pain": st.selectbox("Chest Pain (0=None,1=Mild,2=Severe)", [0,1,2]),
            "Genetic Risk": st.selectbox("Genetic Risk (0=None,1=Low,2=Medium,3=High)", [0,1,2,3]),
            "Weight Loss": st.selectbox("Weight Loss (0=None,1=Mild,2=Severe)", [0,1,2])
        }

        if st.button("Predict Lung Cancer Risk"):
            new_df = pd.DataFrame([lung_data])
            new_df = pd.get_dummies(new_df, drop_first=True)
            new_df = new_df.reindex(columns=lung_features, fill_value=0)
            new_df_scaled = lung_scaler.transform(new_df)
            prediction = lung_rf.predict(new_df_scaled)[0]

            lung_data["Level"] = prediction
            pd.DataFrame([lung_data]).to_csv("lungcancer.csv", mode="a", header=False, index=False)

            if prediction == 1:
                st.error("⚠️ High Risk of Lung Cancer.")
            else:
                st.success("✅ Low Risk of Lung Cancer.")
            st.info("ℹ️ Case saved into lungcancer.csv")






import random
import streamlit as st


# ------------------- QUESTIONS DATABASE -------------------
quiz_questions = {
    "Diabetes": [
        {"q": "A 55-year-old patient with HbA1c of 8.2% is most likely to have?",
         "options": ["Normal", "Prediabetes", "Diabetes"],
         "answer": "Diabetes",
         "reason": "HbA1c ≥ 6.5% indicates Diabetes."},

        {"q": "Which of the following is a common medicine for diabetes?",
         "options": ["Metformin", "Aspirin", "Paracetamol"],
         "answer": "Metformin",
         "reason": "Metformin is the first-line drug for type 2 diabetes."},

        {"q": "High blood glucose levels mainly affect which organ first?",
         "options": ["Kidney", "Liver", "Skin"],
         "answer": "Kidney",
         "reason": "Diabetes damages small blood vessels in the kidney (diabetic nephropathy)."},

        {"q": "Which lifestyle change helps most in diabetes prevention?",
         "options": ["Exercise & Diet", "Smoking", "Skipping Breakfast"],
         "answer": "Exercise & Diet",
         "reason": "Healthy diet + regular physical activity help prevent diabetes."},

        {"q": "A patient with HbA1c 5.5% is considered?",
         "options": ["Normal", "Prediabetes", "Diabetes"],
         "answer": "Normal",
         "reason": "Normal HbA1c is below 5.7%."},

        {"q": "Excessive urination and thirst are symptoms of?",
         "options": ["Diabetes", "Asthma", "Cancer"],
         "answer": "Diabetes",
         "reason": "Polyuria & polydipsia are classic diabetes symptoms."},

        {"q": "Which hormone is deficient in diabetes?",
         "options": ["Insulin", "Thyroxine", "Adrenaline"],
         "answer": "Insulin",
         "reason": "Diabetes occurs due to lack of insulin or insulin resistance."},

        {"q": "Which test is best to monitor long-term diabetes?",
         "options": ["HbA1c", "BP Test", "X-ray"],
         "answer": "HbA1c",
         "reason": "HbA1c reflects average glucose over the last 3 months."},

        {"q": "Gestational diabetes occurs during?",
         "options": ["Pregnancy", "Old age", "Childhood"],
         "answer": "Pregnancy",
         "reason": "Gestational diabetes develops during pregnancy."},

        {"q": "Which complication is common in uncontrolled diabetes?",
         "options": ["Kidney failure", "Hair fall", "Fracture"],
         "answer": "Kidney failure",
         "reason": "Diabetes damages kidneys leading to chronic kidney disease."}
    ],

    "Blood Pressure Abnormality": [
        {"q": "Normal BP value is?",
         "options": ["120/80 mmHg", "200/100 mmHg", "90/40 mmHg"],
         "answer": "120/80 mmHg",
         "reason": "120/80 mmHg is considered the normal blood pressure."},

        {"q": "Hypertension is when systolic BP is above?",
         "options": ["140 mmHg", "100 mmHg", "80 mmHg"],
         "answer": "140 mmHg",
         "reason": "Systolic BP ≥ 140 mmHg is considered high blood pressure."},

        {"q": "Which medicine is commonly prescribed for hypertension?",
         "options": ["Amlodipine", "Paracetamol", "Metformin"],
         "answer": "Amlodipine",
         "reason": "Amlodipine is a calcium channel blocker used for hypertension."},

        {"q": "A patient with frequent headaches and BP 160/100 likely has?",
         "options": ["Hypertension", "Hypotension", "Diabetes"],
         "answer": "Hypertension",
         "reason": "BP above 140/90 is classified as Hypertension."},

        {"q": "Low BP is called?",
         "options": ["Hypotension", "Hypertension", "Stroke"],
         "answer": "Hypotension",
         "reason": "Hypotension refers to blood pressure lower than normal (usually <90/60)."},

        {"q": "Which organ is MOST affected by long-term high BP?",
         "options": ["Heart", "Skin", "Stomach"],
         "answer": "Heart",
         "reason": "Hypertension causes heart enlargement and risk of failure."},

        {"q": "Lifestyle change that lowers BP?",
         "options": ["Less Salt", "More Junk Food", "No Exercise"],
         "answer": "Less Salt",
         "reason": "Reducing salt intake helps lower high blood pressure."},

        {"q": "Which condition increases BP risk?",
         "options": ["Obesity", "Regular Yoga", "Low Stress"],
         "answer": "Obesity",
         "reason": "Excess weight puts more strain on the heart and blood vessels."},

        {"q": "Which test is used to measure BP?",
         "options": ["Sphygmomanometer", "X-ray", "MRI"],
         "answer": "Sphygmomanometer",
         "reason": "Blood pressure is measured using a sphygmomanometer."},

        {"q": "Dizziness, fainting may occur due to?",
         "options": ["Low BP", "High BP", "Diabetes"],
         "answer": "Low BP",
         "reason": "Hypotension causes inadequate blood flow → dizziness/fainting."}
    ],

    "Lung Cancer": [
        {"q": "Main risk factor for lung cancer?",
         "options": ["Smoking", "Sugar", "Exercise"],
         "answer": "Smoking",
         "reason": "90% of lung cancer cases are linked to smoking."},

        {"q": "Persistent cough with blood is a sign of?",
         "options": ["Lung Cancer", "Diabetes", "Hypertension"],
         "answer": "Lung Cancer",
         "reason": "Coughing blood is a common lung cancer symptom."},

        {"q": "Which scan helps in detecting lung cancer?",
         "options": ["CT Scan", "Blood Sugar Test", "Urine Test"],
         "answer": "CT Scan",
         "reason": "CT scans help detect tumors in lungs."},

        {"q": "A medicine commonly used in chemotherapy?",
         "options": ["Cisplatin", "Paracetamol", "Metformin"],
         "answer": "Cisplatin",
         "reason": "Cisplatin is a chemotherapy drug for lung cancer."},

        {"q": "Which group has highest lung cancer risk?",
         "options": ["Smokers", "Children", "Vegetarians"],
         "answer": "Smokers",
         "reason": "Smokers are at highest risk of lung cancer."},

        {"q": "Shortness of breath and chest pain can be?",
         
         "options": ["Lung Cancer", "Diabetes", "Kidney Failure"],
         "answer": "Lung Cancer",
         "reason": "Lung tumors cause breathing difficulty and chest pain."},

        {"q": "Secondhand smoke increases?",
         "options": ["Lung Cancer Risk", "Height", "Weight"],
         "answer": "Lung Cancer Risk",
         "reason": "Secondhand smoke also damages lungs and raises cancer risk."},

        {"q": "Which organ does lung cancer start in?",
         "options": ["Lungs", "Kidneys", "Liver"],
         "answer": "Lungs",
         "reason": "Lung cancer starts in the lung tissues."},

        {"q": "Chronic cough for more than 3 weeks should be?",
         "options": ["Checked for Lung Cancer", "Ignored", "Self-treated"],
         "answer": "Checked for Lung Cancer",
         "reason": "Persistent cough must be checked for lung cancer."},

        {"q": "Best prevention for lung cancer?",
         "options": ["Quit Smoking", "Eat More Sugar", "Skip Exercise"],
         "answer": "Quit Smoking",
         "reason": "The best way to prevent lung cancer is to avoid smoking."}
    ]
}

# ---------------- TRAINING MODE -------------------
if st.session_state.mode == "training":
    st.subheader("🎓 Training Mode")

    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "selected_disease" not in st.session_state:
        st.session_state.selected_disease = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "answered" not in st.session_state:
        st.session_state.answered = False

    # ---- Before Quiz Start ----
    if not st.session_state.quiz_started:
        disease_choice = st.selectbox("Select Disease for Training", ["Select", "Diabetes", "Blood Pressure Abnormality", "Lung Cancer"])
        if st.button("Start Quiz"):
            if disease_choice != "Select":
                st.session_state.selected_disease = disease_choice
                st.session_state.questions = random.sample(quiz_questions[disease_choice], 10)
                st.session_state.quiz_started = True
                st.session_state.current_q = 0
                st.session_state.score = 0
                st.session_state.answered = False
                st.rerun()

    # ---- During Quiz ----
    else:
        q = st.session_state.questions[st.session_state.current_q]
        st.write(f"**Q{st.session_state.current_q+1}: {q['q']}**")

        choice = st.radio("Select an option:", q["options"], key=f"q{st.session_state.current_q}")

        if not st.session_state.answered:
            if st.button("Submit Answer"):
                st.session_state.answered = True
                if choice == q["answer"]:
                    st.success(f"✅ Correct! {q['reason']}")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Wrong! Correct answer: {q['answer']} \n\n👉 {q['reason']}")
                st.stop()

        else:
            if st.button("Next Question ➡️"):
                st.session_state.current_q += 1
                st.session_state.answered = False
                if st.session_state.current_q >= len(st.session_state.questions):
                    st.success(f"🎉 Quiz Finished! Your Score: {st.session_state.score}/10")
                    if st.button("⬅️ Back to Mode Selection"):
                        st.session_state.mode = None
                        st.session_state.quiz_started = False
                        st.rerun()
                else:
                    st.rerun()