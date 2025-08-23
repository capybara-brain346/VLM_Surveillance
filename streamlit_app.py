import os, cv2, sqlite3, time
import pandas as pd
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

# ---------------- CONFIG ----------------
DB_PATH = "vlm_log.db"
TABLE = "caption_logs"
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)
st.set_page_config(page_title="VLM Surveillance", layout="wide")

# ---------------- SESSION STATE ----------------
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None
if "captions" not in st.session_state:
    st.session_state["captions"] = []
if "logs" not in st.session_state:
    st.session_state["logs"] = pd.DataFrame()

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp { background: #0b0f14; color: #e6f2ff; font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { color: #e6f2ff; }
section { padding: 10px 0; margin-bottom: 15px; border-radius: 12px; background: #121821; }
.stButton>button { background: linear-gradient(135deg,#1f6feb,#0ea5e9); color:white; border:none; border-radius:6px; padding:6px 16px; }
.stButton>button:hover { filter: brightness(1.1); }
.dataframe td,.dataframe th { color:#e6f2ff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE}(
            frame TEXT, timestamp TEXT, caption TEXT, location TEXT, alert TEXT
        )""")
init_db()

def insert_log(frame, ts, caption, location="", alert=""):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(f"INSERT INTO {TABLE} VALUES(?,?,?,?,?)", (frame, ts, caption, location, alert))
        con.commit()

def read_logs():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(f"SELECT * FROM {TABLE}", con)

# ---------------- MODELS ----------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------- VIDEO PROCESSING ----------------
def process_video(video_path, frame_skip=20):
    processor, model = load_blip()
    cap = cv2.VideoCapture(video_path)
    frame_count, total = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_skip == 0:
            frame_file = f"{FRAME_DIR}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_file, frame)

            img = Image.open(frame_file).convert("RGB")
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)

            ts = time.strftime("%H:%M:%S")
            insert_log(frame_file, ts, caption)
            total += 1

        frame_count += 1
    cap.release()
    return total

# ---------------- SUMMARY CHUNKING ----------------
def summarize_captions(captions):
    summarizer = load_summarizer()
    text = " ".join(captions)

    max_chunk_len = 800
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_len):
        chunk = " ".join(words[i:i+max_chunk_len])
        chunks.append(chunk)

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    return " ".join(summaries)

# ---------------- Q&A ----------------
def answer_question(question, df):
    q = question.lower()
    if "how many" in q:
        keyword = q.replace("how many", "").strip()
        count = sum(1 for c in df["caption"] if keyword in c.lower())
        return f"{count} {keyword} detected"
    else:
        hits = [(r["timestamp"], r["caption"]) for _, r in df.iterrows() if all(k in r["caption"].lower() for k in q.split())]
        return hits if hits else "No results found"

# ---------------- UI ----------------
st.title("🎥 VLM Surveillance — Analytics Console")

# Upload & Run
st.subheader("Upload & Analyze")
col1, col2 = st.columns([3,1])
with col1:
    video = st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
with col2:
    run = st.button("Run Analysis")

if video:
    # Clear old data when new video uploaded
    if video.name != st.session_state.get("video_name", ""):
        if os.path.exists(DB_PATH): 
            os.remove(DB_PATH)
            init_db()
        if os.path.exists(FRAME_DIR):
            for f in os.listdir(FRAME_DIR):
                os.remove(os.path.join(FRAME_DIR, f))
        st.session_state["video_name"] = video.name

    # Save new video
    video_path = f"uploads_{video.name}"
    with open(video_path,"wb") as f:
        f.write(video.read())
    st.session_state["video_path"] = video_path
    st.video(video_path)

    # Process video
    if run:
        with st.spinner("Processing video..."):
            total = process_video(video_path)
            st.session_state["logs"] = read_logs()
            st.session_state["captions"] = st.session_state["logs"]["caption"].drop_duplicates().tolist()
        st.success(f"Processed {total} frames!")

# Logs Section
st.subheader("📜 Logs")
if not st.session_state["logs"].empty:
    st.dataframe(st.session_state["logs"], use_container_width=True)
else:
    st.info("No logs yet")

# Captions Section
st.subheader("🏷️ Unique Captions")
if st.session_state["captions"]:
    st.dataframe(pd.DataFrame(st.session_state["captions"], columns=["Captions"]), use_container_width=True)
else:
    st.info("No captions yet")

# Summary Section
st.subheader("📝 Summary")
if st.button("Generate Summary") and st.session_state["captions"]:
    summary = summarize_captions(st.session_state["captions"])
    st.success(summary)
elif not st.session_state["captions"]:
    st.info("No captions yet for summary")

# Q&A Section
st.subheader("💬 Q&A")
q = st.text_input("Ask about the video")
if st.button("Ask") and q:
    if not st.session_state["logs"].empty:
        ans = answer_question(q, st.session_state["logs"])
        if isinstance(ans, str): st.warning(ans)
        else: st.dataframe(pd.DataFrame(ans, columns=["Time","Caption"]))
    else:
        st.info("No logs available for Q&A")
