from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def extract_tech_keywords_llm(jd_text):
    prompt = (
        "Extract only technical skills, tools, libraries, platforms, and technologies "
        "from the following job description. Return them as a Python list:\n\n"
        f"{jd_text}\n\nSkills:"
    )
    response = client.text_generation(prompt, max_new_tokens=300, temperature=0.3, do_sample=True)
    # attempt to parse response into list
    try:
        import ast
        return set(ast.literal_eval(response.strip()))
    except:
        # fallback: split on commas
        return set(map(str.strip, response.strip().split(',')))

def generate_questions_llama(jd_text):
    prompt = f"Generate 5 technical interview questions for this job description:\n\n{jd_text}"
    response = client.text_generation(prompt, max_new_tokens=300, temperature=0.7, return_full_text=False)
    return response.strip()

def assess_answer_llama(user_answer):
    prompt = f"Rate this interview answer and give feedback:\n\n{user_answer}"
    response = client.text_generation(prompt, max_new_tokens=300, temperature=0.7, return_full_text=False)
    return response.strip()
