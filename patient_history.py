# patient_history.py
import psycopg2
from datetime import datetime
import streamlit as st

# 🔗 Database connection
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="medical_ai",
        user="postgres",
        password="12345678", 
        port=5432
    )

# 🏗️ Initialize the table (run once during app startup)
def init_patient_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            patient_name VARCHAR(100),
            age INTEGER,
            gender VARCHAR(10),
            symptoms TEXT,
            disease VARCHAR(50),
            diagnosis_result VARCHAR(50),
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

# 💾 Save patient details & diagnosis result
def save_patient_record(user_id, patient_data, disease, result, confidence):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO patient_records (
                user_id, patient_name, age, gender, symptoms, disease, diagnosis_result, confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            patient_data.get("Name", "Unknown"),
            patient_data.get("Age", 0),
            patient_data.get("Sex", "Unknown"),
            ", ".join(patient_data.get("Symptoms", [])) if isinstance(patient_data.get("Symptoms"), list) else str(patient_data.get("Symptoms", "")),
            disease,
            result,
            confidence
        ))
        conn.commit()
        st.success("📦 Patient record saved successfully.")
    except Exception as e:
        st.error(f"❌ Error saving patient record: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

# 📜 Fetch recent records for the current user
def get_patient_records(user_id, limit=5):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_name, age, gender, symptoms, disease, diagnosis_result, confidence_score, created_at
        FROM patient_records
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (user_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# 🧾 Display records inside Streamlit
def display_patient_records(user_id):
    st.markdown("### 📋 Previous Patient Records")
    records = get_patient_records(user_id)
    if records:
        for r in records:
            st.write(f"🧍 **{r[0]}**, {r[1]} yrs, {r[2]}")
            st.write(f"💬 Symptoms: {r[3]}")
            st.write(f"🦠 Disease: {r[4]} | 🩺 Result: {r[5]} ({r[6]:.1f}% confidence)")
            st.write(f"🕒 Date: {r[7]}")
            st.markdown("---")
    else:
        st.info("No previous records found.")
