import streamlit as st
from parser import extract_text_from_pdf
from matcher import compute_similarity, keyword_difference
from interview import generate_questions_llama, assess_answer_llama

st.title("CV ↔ JD Matcher + AI Feedback")

uploaded_cv = st.file_uploader("Upload your CV (PDF)", type="pdf")
uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type="pdf")

if uploaded_cv and uploaded_jd:
    cv_text = extract_text_from_pdf(uploaded_cv)
    jd_text = extract_text_from_pdf(uploaded_jd)
    match_score = compute_similarity(cv_text, jd_text)

    st.subheader(f"Match Score: {match_score}%")
    if match_score < 60:
        st.warning("Low alignment. Improve keyword match.")

    missing_keywords, matched_keywords = keyword_difference(cv_text, jd_text)

    st.subheader("Keyword Analysis")
    if missing_keywords:
        st.markdown("❌ **Missing Keywords from CV:**")
        st.write(", ".join(missing_keywords[:20]))

        st.markdown("💡 **Suggestions:**")
        for kw in missing_keywords[:5]:
            st.write(f"Add '{kw}' where relevant (projects, skills, etc.)")

    else:
        st.success("Your CV covers most keywords.")

    if st.button("Generate Interview Questions"):
        questions = generate_questions_llama(jd_text)
        st.text_area("Sample Interview Questions", questions, height=200)

    user_ans = st.text_area("Paste your answer below for feedback")
    if user_ans:
        feedback = assess_answer_llama(user_ans)
        st.text_area("AI Feedback", feedback, height=200)
