

#from VLM_Surveillance.main import fetch_logs
from main import fetch_logs

from transformers import pipeline
import sqlite3
import os
import google.generativeai as genai
from dotenv import load_dotenv
# ========== 1. VIDEO SUMMARY ==========
def generate_summary(logs):
    captions = " ".join([log[2] for log in logs if log[2]])  # log[2] = caption
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(captions, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    return summary

# ========== 2. SEARCH FUNCTION ==========
def search_logs(logs, keyword=None, after=None):
    result = []
    for log in logs:
        if keyword and keyword.lower() not in log[2].lower():
            continue
        if after and log[1] <= after:
            continue
        result.append(log)
    return result

# ========== 3. NATURAL LANGUAGE TO SQL using Gemini 1.5 Flash ==========
def answer_question_gemini_flash(nl_question, db_path="vlm_log.db"):
    import google.generativeai as genai
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("models/gemini-1.5-flash")

    schema = """
    Table: caption_logs
    Columns:
      id INTEGER
      frame TEXT
      timestamp TEXT
      caption TEXT
      location TEXT
      alert TEXT
    """

    prompt = f"""
    Write only a clean SQL query for the question below.
    No markdown, no explanations, no prefix like "sql".
    Just output the SQL statement.

    Schema:
    {schema}

    Question:
    {nl_question}
    """

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Sanitize output
    lines = raw.splitlines()
    sql_lines = [line for line in lines if line.strip().lower().startswith("select") or line.strip().lower().startswith("with")]
    sql_query = sql_lines[0].strip() if sql_lines else raw

    #print("🔎 Cleaned SQL:", sql_query)

    # Execute the query
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
    except Exception as e:
        result = [f"❌ SQL Error: {e}"]
    conn.close()
    return result

# ========== CLI ==========
if __name__ == "__main__":
    logs = fetch_logs()

    print("\n=== VIDEO SUMMARY ===")
    print(generate_summary(logs))

    print("\n=== SEARCH ===")
    keyword = input("Search keyword (e.g. truck/loitering): ")
    filtered = search_logs(logs, keyword=keyword)
    for log in filtered:
        print(log)

   
    print("\n=== NATURAL LANGUAGE Q&A (Gemini 1.5 Flash) ===")
    q = input("Ask a question (e.g. What trucks entered after noon?): ")
    results = answer_question_gemini_flash(q)
    unique_answers = sorted(set([r[0] for r in results if r and r[0]]))
    if unique_answers:
        #print("📍 Answer:", ", ".join(unique_answers))
        print("📍 Answer:", ", ".join(str(ans) for ans in unique_answers))

    else:
        print("❌ No result.")

