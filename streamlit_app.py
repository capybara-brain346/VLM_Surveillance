import os
import cv2
import sqlite3
import time
import threading
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import base64
import numpy as np
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from io import BytesIO


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
if "realtime_active" not in st.session_state:
    st.session_state["realtime_active"] = False
if "realtime_thread" not in st.session_state:
    st.session_state["realtime_thread"] = None
if "live_frame" not in st.session_state:
    st.session_state["live_frame"] = None
if "detection_count" not in st.session_state:
    st.session_state["detection_count"] = 0
if "windows_server_url" not in st.session_state:
    st.session_state["windows_server_url"] = "http://192.168.1.100:5000"
if "server_connected" not in st.session_state:
    st.session_state["server_connected"] = False

# ---------------- STYLING ----------------
st.markdown(
    """
<style>
.stApp { background: #0b0f14; color: #e6f2ff; font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { color: #e6f2ff; }
section { padding: 10px 0; margin-bottom: 15px; border-radius: 12px; background: #121821; }
.stButton>button { background: linear-gradient(135deg,#1f6feb,#0ea5e9); color:white; border:none; border-radius:6px; padding:6px 16px; }
.stButton>button:hover { filter: brightness(1.1); }
.dataframe td,.dataframe th { color:#e6f2ff !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE}(
            frame TEXT, timestamp TEXT, caption TEXT, location TEXT, alert TEXT,
            confidence REAL, source TEXT, detection_type TEXT
        )""")


init_db()


def insert_log(
    frame,
    ts,
    caption,
    location="",
    alert="",
    confidence=0.0,
    source="video",
    detection_type="caption",
):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            f"INSERT INTO {TABLE} VALUES(?,?,?,?,?,?,?,?)",
            (frame, ts, caption, location, alert, confidence, source, detection_type),
        )
        con.commit()


def read_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(f"SELECT * FROM {TABLE}", con)


# ---------------- MODELS ----------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


# ---------------- WEBCAM SERVER FUNCTIONS ----------------
def check_server_connection(server_url):
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_windows_webcam():
    try:
        response = requests.get(
            f"{st.session_state['windows_server_url']}/start", timeout=10
        )
        return response.status_code == 200
    except:
        return False


def stop_windows_webcam():
    try:
        response = requests.get(
            f"{st.session_state['windows_server_url']}/stop", timeout=5
        )
        return response.status_code == 200
    except:
        return False


def get_frame_from_server():
    try:
        response = requests.get(
            f"{st.session_state['windows_server_url']}/frame", timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                frame_data = base64.b64decode(data["frame"])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
    except:
        pass
    return None


# ---------------- REAL-TIME PROCESSING ----------------
def real_time_processing():
    processor, model = load_blip()
    frame_count = 0

    while st.session_state["realtime_active"]:
        frame = get_frame_from_server()

        if frame is None:
            time.sleep(0.5)
            continue

        if frame_count % 30 == 0:
            frame_file = f"{FRAME_DIR}/live_frame_{int(time.time())}.jpg"
            cv2.imwrite(frame_file, frame)

            img = Image.open(frame_file).convert("RGB")
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_log(
                frame_file,
                ts,
                caption,
                source="live_camera",
                detection_type="real_time",
            )

            st.session_state["live_frame"] = frame_file
            st.session_state["detection_count"] += 1

        frame_count += 1
        time.sleep(0.1)


# ---------------- VIDEO PROCESSING ----------------
def process_video(video_path, frame_skip=20):
    processor, model = load_blip()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_file = f"{FRAME_DIR}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_file, frame)

            img = Image.open(frame_file).convert("RGB")
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_log(
                frame_file,
                ts,
                caption,
                source="video",
                detection_type="batch_processing",
            )
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
        chunk = " ".join(words[i : i + max_chunk_len])
        chunks.append(chunk)

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0][
            "summary_text"
        ]
        summaries.append(summary)

    return " ".join(summaries)


# ---------------- STATISTICAL DASHBOARD ----------------
def get_detection_stats():
    df = read_logs()
    if df.empty:
        return None

    stats = {
        "total_detections": len(df),
        "unique_captions": df["caption"].nunique(),
        "sources": df["source"].value_counts().to_dict()
        if "source" in df.columns
        else {},
        "hourly_activity": [],
        "top_detections": [],
    }

    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        stats["hourly_activity"] = df["hour"].value_counts().sort_index().to_dict()

    if "caption" in df.columns:
        stats["top_detections"] = df["caption"].value_counts().head(10).to_dict()

    return stats


def create_activity_timeline():
    df = read_logs()
    if df.empty or "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="detections")

    fig = px.line(
        daily_counts,
        x="date",
        y="detections",
        title="Daily Detection Activity",
        color_discrete_sequence=["#1f6feb"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6f2ff",
    )
    return fig


def create_hourly_heatmap():
    df = read_logs()
    if df.empty or "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()

    heatmap_data = df.groupby(["day", "hour"]).size().unstack(fill_value=0)

    fig = px.imshow(
        heatmap_data,
        title="Activity Heatmap (Day vs Hour)",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6f2ff",
    )
    return fig


def create_detection_pie():
    df = read_logs()
    if df.empty or "caption" not in df.columns:
        return None

    top_detections = df["caption"].value_counts().head(8)

    fig = px.pie(
        values=top_detections.values,
        names=top_detections.index,
        title="Top Detection Types",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6f2ff",
    )
    return fig


# ---------------- Q&A ----------------
def answer_question(question, df):
    q = question.lower()
    if "how many" in q:
        keyword = q.replace("how many", "").strip()
        count = sum(1 for c in df["caption"] if keyword in c.lower())
        return f"{count} {keyword} detected"
    else:
        hits = [
            (r["timestamp"], r["caption"])
            for _, r in df.iterrows()
            if all(k in r["caption"].lower() for k in q.split())
        ]
        return hits if hits else "No results found"


# ---------------- UI ----------------
st.title("🎥 VLM Surveillance — Analytics Console")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📹 Video Analysis", "📡 Live Monitoring", "📊 Dashboard", "💬 Q&A"]
)

with tab1:
    st.subheader("Upload & Analyze")
    col1, col2 = st.columns([3, 1])
    with col1:
        video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
    with col2:
        run = st.button("Run Analysis")

    if video:
        if video.name != st.session_state.get("video_name", ""):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
                init_db()
            if os.path.exists(FRAME_DIR):
                for f in os.listdir(FRAME_DIR):
                    os.remove(os.path.join(FRAME_DIR, f))
            st.session_state["video_name"] = video.name

        video_path = f"uploads_{video.name}"
        with open(video_path, "wb") as f:
            f.write(video.read())
        st.session_state["video_path"] = video_path
        st.video(video_path)

        if run:
            with st.spinner("Processing video..."):
                total = process_video(video_path)
                st.session_state["logs"] = read_logs()
                st.session_state["captions"] = (
                    st.session_state["logs"]["caption"].drop_duplicates().tolist()
                )
            st.success(f"Processed {total} frames!")

    st.subheader("📜 Logs")
    if not st.session_state["logs"].empty:
        st.dataframe(st.session_state["logs"], use_container_width=True)
    else:
        st.info("No logs yet")

    st.subheader("🏷️ Unique Captions")
    if st.session_state["captions"]:
        st.dataframe(
            pd.DataFrame(st.session_state["captions"], columns=["Captions"]),
            use_container_width=True,
        )
    else:
        st.info("No captions yet")

    st.subheader("📝 Summary")
    if st.button("Generate Summary") and st.session_state["captions"]:
        summary = summarize_captions(st.session_state["captions"])
        st.success(summary)
    elif not st.session_state["captions"]:
        st.info("No captions yet for summary")

with tab2:
    st.subheader("📡 Live Camera Monitoring")

    # Server Configuration
    st.subheader("🔧 Windows Server Configuration")
    with st.expander(
        "Configure Windows Webcam Server",
        expanded=not st.session_state["server_connected"],
    ):
        st.info(
            "**Instructions:** Run `windows_webcam_server.py` on your Windows machine first!"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            server_url = st.text_input(
                "Windows Server URL",
                value=st.session_state["windows_server_url"],
                help="Format: http://WINDOWS_IP:5000 (find your Windows IP by running the server)",
            )
        with col2:
            if st.button("Test Connection"):
                if check_server_connection(server_url):
                    st.session_state["windows_server_url"] = server_url
                    st.session_state["server_connected"] = True
                    st.success("✅ Connected!")
                else:
                    st.session_state["server_connected"] = False
                    st.error("❌ Connection failed")

        if st.session_state["server_connected"]:
            st.success(f"✅ Connected to: {st.session_state['windows_server_url']}")
        else:
            st.warning(
                "⚠️ Server not connected. Please start the Windows server and test connection."
            )

    # Live Monitoring Controls
    st.subheader("🎥 Live Monitoring")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        if st.button(
            "🟢 Start Live Feed",
            disabled=st.session_state["realtime_active"]
            or not st.session_state["server_connected"],
        ):
            if start_windows_webcam():
                st.session_state["realtime_active"] = True
                st.session_state["realtime_thread"] = threading.Thread(
                    target=real_time_processing, daemon=True
                )
                st.session_state["realtime_thread"].start()
                st.success("Live monitoring started!")
            else:
                st.error("Failed to start Windows webcam server")

    with col2:
        if st.button(
            "🔴 Stop Live Feed", disabled=not st.session_state["realtime_active"]
        ):
            st.session_state["realtime_active"] = False
            stop_windows_webcam()
            st.success("Live monitoring stopped!")

    with col3:
        st.metric("Live Detections", st.session_state["detection_count"])

    with col4:
        connection_status = (
            "🟢 Connected"
            if st.session_state["server_connected"]
            else "🔴 Disconnected"
        )
        st.metric("Server Status", connection_status)

    if st.session_state["realtime_active"]:
        st.info("🔴 LIVE - Camera feed is active")
        if st.session_state["live_frame"] and os.path.exists(
            st.session_state["live_frame"]
        ):
            st.image(
                st.session_state["live_frame"],
                caption="Latest Detection",
                use_column_width=True,
            )

    st.subheader("Recent Live Detections")
    live_logs = read_logs()
    if not live_logs.empty and "source" in live_logs.columns:
        live_data = live_logs[live_logs["source"] == "live_camera"].tail(10)
        if not live_data.empty:
            st.dataframe(live_data, use_container_width=True)
        else:
            st.info("No live detections yet")
    else:
        st.info("No live detections yet")

with tab3:
    st.subheader("📊 Statistical Dashboard")

    stats = get_detection_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Detections", stats["total_detections"])
        with col2:
            st.metric("Unique Captions", stats["unique_captions"])
        with col3:
            video_count = stats["sources"].get("video", 0)
            st.metric("Video Detections", video_count)
        with col4:
            live_count = stats["sources"].get("live_camera", 0)
            st.metric("Live Detections", live_count)

        col1, col2 = st.columns(2)

        with col1:
            timeline_fig = create_activity_timeline()
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)

            pie_fig = create_detection_pie()
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)

        with col2:
            heatmap_fig = create_hourly_heatmap()
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info(
            "No data available for dashboard. Process a video or start live monitoring first."
        )

with tab4:
    st.subheader("💬 Q&A")
    q = st.text_input("Ask about the surveillance data")
    if st.button("Ask") and q:
        current_logs = read_logs()
        if not current_logs.empty:
            ans = answer_question(q, current_logs)
            if isinstance(ans, str):
                st.warning(ans)
            else:
                st.dataframe(pd.DataFrame(ans, columns=["Time", "Caption"]))
        else:
            st.info("No logs available for Q&A")
