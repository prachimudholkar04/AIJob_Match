import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from interview import extract_tech_keywords_llm


nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(cv_text, jd_text):
    emb_cv = model.encode(cv_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(emb_cv, emb_jd).item()
    return round(score * 100, 2)

def extract_keywords(text):
    doc = nlp(text)
    noun_chunks = {chunk.text.lower().strip() for chunk in doc.noun_chunks}
    filtered = {
        phrase for phrase in noun_chunks
        if len(phrase) > 2 and phrase not in ENGLISH_STOP_WORDS
    }
    return filtered

def keyword_difference(cv_text, jd_text):
    jd_tech = extract_tech_keywords_llm(jd_text)
    cv_tech = extract_tech_keywords_llm(cv_text)

    missing = jd_tech - cv_tech
    matched = cv_tech & jd_tech
    return sorted(missing), sorted(matched)