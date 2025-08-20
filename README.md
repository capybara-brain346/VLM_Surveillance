

# VLM Based Surveillance enhanced with summarizer and queries

---

## 🛰️ **Problem Statement**

Design an AI-powered **Drone Security Analyst Agent** that:

* Analyzes live drone **video** and **telemetry**
* Identifies **objects** (people, cars, etc.) and **events** (loitering, crash)
* Logs them with **context** (e.g., time, location)
* Generates real-time **alerts**
* Allows **video indexing** and **natural language querying**
* Uses **AI tools** where applicable (Gemini, BLIP, BART)

---

## ✅ **Solution Summary**

This solution includes:

1. Frame extraction + captioning with **BLIP**
2. Logging with alert detection + SQLite indexing
3. Caption summarization using **facebook/bart-large-cnn**
4. Natural Language Q\&A using **Gemini 1.5 Flash**
5. Secure API handling with `.env`

The result is a robust CLI tool that can log, summarize, and answer surveillance-related questions on any video stream.

---

## 🧠 **Architecture Overview**

```
Video → Frames → BLIP (captioning) → Parse → Store in SQLite
                          ↘
                    Detect alerts (loitering, crash, etc.)

Logs → Summarizer (BART)
     → Gemini Q&A (natural language → SQL)
```

---

## 🧩 **Components Used**

### 🔍 1. Visual Language Model – BLIP

* Model: `Salesforce/blip-image-captioning-base`
* Task: Converts each frame into a text description
* Example Output:
  `"a car is parked in a garage"`

### 📊 2. Caption Parser

* Extracts keywords from captions:

  * `vehicle`, `person`, `action`, `location`
* Also checks for **alert keywords**:

  * `["loitering", "crash", "assault", "fight", "harm", "accident"]`

### 🗃️ 3. SQLite Logging

* Captions and alerts are stored in `vlm_log.db`
* Schema:

  ```
  id, frame, timestamp, caption, location, alert
  ```

### 🧠 4. Summarization (facebook/bart-large-cnn)

* Merges all captions into a concise summary
* Example:

  ```
  "A man is seen near a tunnel. A car is parked in a garage."
  ```

### 💬 5. Gemini 1.5 Flash – Natural Language to SQL

* User types: `"Where was the car seen?"`
* Gemini converts to:

  ```sql
  SELECT location FROM caption_logs WHERE caption LIKE '%car%'
  ```
* Answer: `"parking"`

---

## 💻 **Scripts**

### `main.py` – Frame Analysis + Logging

* Reads video (`fdoor.mp4`)
* Extracts frames every N seconds
* Captions frames using BLIP
* Parses + stores captions in DB
* Detects and logs alerts

### `app.py` – Summary + Search + Gemini Q\&A

* Loads logs from DB
* Generates one-line summary of all events
* Supports:

  * Keyword search (e.g., “truck”)
  * Natural questions (via Gemini)

---

## 🧪 **Test Examples**

| Input                          | Output                                                                  |
| ------------------------------ | ----------------------------------------------------------------------- |
| `"Where was car seen?"`        | `📍 Answer: parking`                                                    |
| `"What frames had loitering?"` | `📍 Answer: frame_0012.jpg, frame_0015.jpg`                             |
| `"Summarize the video"`        | `"A motorcycle drove through a tunnel and a man was seen at the gate."` |

---

## 🔐 **API + Environment Setup**

### Environment Variables (`.env`):

```
GEMINI_API_KEY=your_api_key_here
```

### Loading:

```python
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

---

## 🗃️ **Project Structure**

```
VLM_Surveillance/
├── main.py           # Frame logging with BLIP
├── app.py            # Summary + Gemini Q&A
├── vlm_log.db        # Indexed captions
├── frames/           # Extracted frames
├── .env              # (not committed) Gemini key
├── .env.example      # Template
```

---

## 📚 **AI Tools Used**

| Tool             | Use                      |
| ---------------- | ------------------------ |
| BLIP             | Image captioning         |
| BART             | Video summarization      |
| Gemini 1.5 Flash | Natural language → SQL   |
| Hugging Face     | Model loading            |
| OpenCV           | Frame extraction         |
| SQLite           | Local DB for frame index |

---

## ✅ **Features Implemented**

* [x] Real-time video captioning
* [x] Alert detection for security keywords
* [x] Timestamped event logging
* [x] Caption summarization
* [x] Keyword-based search
* [x] Natural question answering via Gemini

---
