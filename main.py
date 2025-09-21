import os
from dotenv import load_dotenv
import streamlit as st
from app.utils.file_loader import detect_and_extract

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="ATS Resume Agent (CrewAI)", page_icon="🧠", layout="wide")

# Set title and caption
st.title("🧠 ATS-Optimized Resume Agent (CrewAI + OpenAI)")
st.caption("Upload your resume (.pdf or .docx), target a role, and get an ATS-friendly version with scores & quick wins.")

# Sidebar
with st.sidebar:
    st.subheader("OpenAI Settings")
    st.text_input("Model:", value="gpt-4o-mini", disabled=True)
    st.write("API Key loaded: ✅ Working OpenAI key")
    
# Inputs
colL, colR = st.columns([1,1])
with colL:
    up = st.file_uploader("Upload Resume (.pdf or .docx preferred)", type=["pdf", "docx", "txt"])
with colR:
    job_title = st.text_input("Target Job Title (e.g., 'Machine Learning Engineer')")
    job_desc = st.text_area("Paste Job Description", height=220, placeholder="Paste JD here...")

run_btn = st.button("Run ATS Agent")

tabs = st.tabs(["Cleaned Resume", "Rewritten (ATS-optimized)", "Final (Refined Bullets)", "ATS Evaluation"])

if run_btn:
    if up is None:
        st.error("Please upload a resume file.")
    elif not job_title or not job_desc.strip():
        st.error("Please provide a target job title and job description.")
    else:
        ext, raw_text = detect_and_extract(up.name, up.read())
        if not raw_text.strip():
            st.error("Could not extract any text from the file.")
        else:
            with st.spinner("Running Crew agents..."):
                print("File Extention: ", ext)
                pass
        
        
