"""
SAHAYAK — Multilingual Accessibility Reader
An AI agent with four modes:
  1. Document Reader   - photo of a form/notice -> simplified, translated, read aloud
  2. Text Message       - type/paste text -> simplified, translated, read aloud
  3. Sign & Menu Reader - photo of a sign/menu/board -> translated, read aloud
  4. Voice Conversation - speak into the mic -> transcribed, translated, read aloud

Run with:  streamlit run app.py
"""

import os
import json
import io
import hmac
import datetime
import streamlit as st
from PIL import Image
import pytesseract
from gtts import gTTS
import google.generativeai as genai
import speech_recognition as sr
import fitz  # PyMuPDF, for reading PDFs
import qrcode

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Set your FREE Gemini API key(s) as environment variables before running:
#   Primary:  export GEMINI_API_KEY="your-key-here"        (Mac/Linux)
#             setx GEMINI_API_KEY "your-key-here"           (Windows)
#   Backup:   export GEMINI_API_KEY_2="backup-key-here"     (Mac/Linux)
#             setx GEMINI_API_KEY_2 "backup-key-here"       (Windows)
# The backup key is optional — get free keys at: https://aistudio.google.com
def get_key(name: str):
    """
    Look up an API key two ways so the same code works both locally and on
    Streamlit Community Cloud:
      1. Streamlit Cloud's secrets manager (st.secrets)
      2. A local environment variable (setx / export)
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml present locally — that's fine
    return os.environ.get(name)


PRIMARY_KEY = get_key("GEMINI_API_KEY")
BACKUP_KEY = get_key("GEMINI_API_KEY_2")  # optional, from a second Google account

genai.configure(api_key=PRIMARY_KEY)
model = genai.GenerativeModel("gemini-3.6-flash")


def generate_with_fallback(prompt: str):
    """Call the model with the primary key; if its free-tier quota is
    exhausted, automatically switch to the backup key (if one is set)
    and retry once."""
    from google.api_core.exceptions import ResourceExhausted

    try:
        genai.configure(api_key=PRIMARY_KEY)
        return model.generate_content(prompt)
    except ResourceExhausted:
        if not BACKUP_KEY:
            raise
        st.toast("Primary key hit its daily limit — switching to backup key.", icon="🔁")
        genai.configure(api_key=BACKUP_KEY)
        return model.generate_content(prompt)

LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
}

# Speech-recognition language codes (for the voice mode)
SPEECH_LANGUAGES = {
    "English": "en-IN",
    "Telugu": "te-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
}

# Predefined emergency phrases, pre-translated so this mode works instantly
# with NO API call — important for poor-connectivity / urgent situations.
# NOTE: these translations were written by the AI assistant and have not
# been verified by a native speaker — double-check before relying on them.
EMERGENCY_PHRASES = {
    "🔥 Fire": {
        "English": "There is a fire! Please send the fire department immediately.",
        "Telugu": "మంటలు వచ్చాయి! దయచేసి వెంటనే అగ్నిమాపక దళాన్ని పంపండి.",
        "Hindi": "आग लगी है! कृपया तुरंत दमकल विभाग को भेजें।",
        "Tamil": "தீ பிடித்துவிட்டது! தயவுசெய்து உடனடியாக தீயணைப்புத் துறையை அனுப்பவும்.",
    },
    "🚑 Medical": {
        "English": "This is a medical emergency. We need an ambulance right now.",
        "Telugu": "ఇది వైద్య అత్యవసర పరిస్థితి. మాకు వెంటనే అంబులెన్స్ కావాలి.",
        "Hindi": "यह एक चिकित्सा आपातकाल है। हमें अभी एम्बुलेंस चाहिए।",
        "Tamil": "இது ஒரு மருத்துவ அவசரநிலை. எங்களுக்கு உடனே ஆம்புலன்ஸ் தேவை.",
    },
    "🚔 Police": {
        "English": "We need police help immediately. There is an emergency.",
        "Telugu": "మాకు వెంటనే పోలీసుల సహాయం కావాలి. ఇది అత్యవసర పరిస్థితి.",
        "Hindi": "हमें तुरंत पुलिस की मदद चाहिए। यह एक आपातकालीन स्थिति है।",
        "Tamil": "எங்களுக்கு உடனடியாக காவல்துறை உதவி தேவை. இது ஒரு அவசரநிலை.",
    },
    "🚨 Accident": {
        "English": "There has been an accident. We need help right away.",
        "Telugu": "ఒక ప్రమాదం జరిగింది. మాకు వెంటనే సహాయం కావాలి.",
        "Hindi": "एक दुर्घटना हुई है। हमें तुरंत मदद चाहिए।",
        "Tamil": "ஒரு விபத்து நடந்துள்ளது. எங்களுக்கு உடனே உதவி தேவை.",
    },
    "🆘 Lost Person": {
        "English": "I am lost. I need help finding my way.",
        "Telugu": "నేను దారి తప్పిపోయాను. నాకు సహాయం కావాలి.",
        "Hindi": "मैं रास्ता भटक गया हूं। मुझे मदद चाहिए।",
        "Tamil": "நான் வழி தவறிவிட்டேன். எனக்கு உதவி தேவை.",
    },
}

# ---------------------------------------------------------------------
# CORE AGENT FUNCTIONS
# ---------------------------------------------------------------------

def extract_text_from_image(image: Image.Image) -> tuple[str, float]:
    """Run OCR on an image. Returns (text, confidence_estimate)."""
    raw_text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if c != "-1"]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return raw_text.strip(), avg_conf


def simplify_and_analyze(raw_text: str) -> dict:
    """Simplify text and flag urgent info. Returns strict JSON."""
    prompt = f"""You are an accessibility assistant. A user has scanned a document
and needs it explained simply. Here is the raw extracted text:

---
{raw_text}
---

Return ONLY valid JSON (no markdown fences, no extra text) with this exact shape:
{{
  "simplified_text": "the document rewritten in short, plain sentences a beginner reader can understand",
  "has_urgent_info": true or false,
  "urgent_summary": "one short sentence naming the deadline/amount/warning if any, else empty string",
  "reading_level_used": "simple" or "very_simple"
}}"""
    response = generate_with_fallback(prompt)
    text_out = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text_out)


def translate_text(text: str, target_language: str, source_hint: str = "English") -> str:
    """Translate text into the target language."""
    if target_language == source_hint:
        return text

    prompt = f"""Translate the following text into {target_language}.
Keep it natural and simple, as if explaining to someone reading it for the first time.
Return ONLY the translated text, nothing else.

Text:
{text}"""
    response = generate_with_fallback(prompt)
    return response.text.strip()


TONE_LABELS = {
    "neutral": "😐 Neutral",
    "friendly": "😊 Friendly",
    "angry": "😡 Angry / Urgent",
    "sad": "😢 Sad",
    "excited": "🎉 Excited",
    "formal": "📢 Formal",
}


def detect_and_translate_tone(text: str, target_language: str, source_hint: str = "English") -> dict:
    """
    Detect the likely communication tone of the text (a surface-level read
    of word choice and punctuation, NOT a real emotional/psychological
    assessment of the speaker) and translate it while preserving that tone
    rather than flattening everything into neutral, formal language.
    """
    prompt = f"""You are a translation assistant that pays attention to tone.

Step 1: Read the text below and classify its communication tone as ONE of:
neutral, friendly, angry, sad, excited, formal
(base this only on word choice, punctuation, and phrasing — not a real
psychological judgment of the person)

Step 2: Translate the text into {target_language}, keeping that same tone
and level of urgency/warmth in the translation — don't flatten an urgent
or angry message into a calm, neutral one.

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{"tone": "one of the 6 words above", "translated_text": "the translation"}}

Text:
{text}"""
    response = generate_with_fallback(prompt)
    text_out = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(text_out)
    result["tone_label"] = TONE_LABELS.get(result.get("tone", "neutral"), "😐 Neutral")
    return result


def text_to_speech(text: str, lang_code: str, filename: str = "output.mp3") -> str:
    """Convert text to an audio file using gTTS."""
    tts = gTTS(text=text, lang=lang_code)
    tts.save(filename)
    return filename


def transcribe_audio(audio_bytes: bytes, lang_code: str) -> str:
    """Transcribe recorded audio (WAV bytes) to text using Google's free
    speech recognition API."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language=lang_code)


def pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    """Convert the first page of a PDF into a PIL Image for OCR."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def make_qr_code(text: str) -> Image.Image:
    """
    Generate a QR code that encodes the given text directly (not a link) —
    this works with zero hosting/internet needed to scan it, but is limited
    to text content only (not audio or images) due to QR data capacity.
    Indic-script text (Telugu/Hindi/Tamil) takes ~3 bytes/character in
    UTF-8, so we cap length more conservatively and use a higher error
    correction level so it still scans reliably on a phone camera.
    """
    MAX_CHARS = 350
    truncated = len(text) > MAX_CHARS
    data = text[:MAX_CHARS]

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=3)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0D47A1", back_color="white").convert("RGB")
    return img, truncated


def render_qr_share_button(content: str, key: str, label: str = "📱 Generate QR code to share"):
    """Render a button that, when clicked, shows a QR code encoding the given text."""
    if st.button(label, key=key):
        if not content.strip():
            st.warning("Nothing to share yet.")
        else:
            try:
                qr_img, was_truncated = make_qr_code(content)
                st.image(qr_img, caption="Scan to read this text on another phone", width=260)
                if was_truncated:
                    st.caption(
                        "⚠️ Text was long, so only the first part is in this QR code "
                        "(QR codes hold a limited amount of data)."
                    )
                st.caption(
                    "This QR code encodes the translated text directly (works offline, "
                    "no hosting needed) — it can't carry the audio or original photo."
                )
            except Exception as e:
                st.error(f"Couldn't generate a QR code for this text: {e}")


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Load a PIL Image from an uploaded file, handling both image and PDF types."""
    filename = getattr(uploaded_file, "name", "") or ""
    if filename.lower().endswith(".pdf"):
        return pdf_to_image(uploaded_file.getvalue())
    return Image.open(uploaded_file)


def add_to_history(mode: str, original: str, simplified: str, language: str):
    """Append an entry to the session's reading history."""
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "time": datetime.datetime.now().strftime("%I:%M %p"),
        "mode": mode,
        "original": original[:200],
        "result": simplified[:300],
        "language": language,
    })
    # Keep only the most recent 20 entries
    st.session_state.history = st.session_state.history[:20]


def make_app_icon() -> Image.Image:
    """Generate a simple circular logo (blue circle with a white speech-bubble
    mark) so the app has its own icon instead of a generic emoji."""
    from PIL import ImageDraw

    size = 128
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.ellipse((4, 4, size - 4, size - 4), fill=(124, 58, 237, 255))  # violet circle
    # simple speech-bubble mark
    draw.rounded_rectangle((32, 34, 96, 78), radius=14, fill=(255, 255, 255, 255))
    draw.polygon([(46, 76), (46, 96), (66, 76)], fill=(255, 255, 255, 255))
    return icon


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="SAHAYAK", page_icon=make_app_icon(), layout="centered")

BASE_THEME_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(124, 58, 237, 0.16) 0%, transparent 42%),
            radial-gradient(circle at 92% 12%, rgba(6, 182, 212, 0.16) 0%, transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.10) 0%, transparent 45%),
            linear-gradient(180deg, #F7F5FF 0%, #FFFFFF 45%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(200deg, #6D28D9 0%, #4338CA 45%, #0891B2 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #F5F3FF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.10) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }

    h1 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        color: #3B0764 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h1::after {
        content: "";
        display: block;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #7C3AED, #06B6D4);
        border-radius: 2px;
        margin-top: 8px;
    }
    h2, h3 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        color: #4C1D95 !important;
        font-weight: 600 !important;
    }
    p, span, div, label { letter-spacing: 0.1px; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(124, 58, 237, 0.07);
        backdrop-filter: blur(8px);
        padding: 6px;
        border-radius: 999px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        color: #4C1D95;
        transition: background-color 0.2s ease, color 0.2s ease, transform 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7C3AED, #06B6D4) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
    }

    .stButton button {
        background: linear-gradient(90deg, #7C3AED 0%, #6366F1 55%, #06B6D4 100%);
        color: white;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.55em 1.4em;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .stButton button:hover {
        filter: brightness(1.08);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45);
    }
    .stButton button:active {
        transform: translateY(0px);
    }

    /* Glass-card look for expanders, containers, and file uploaders —
       purely cosmetic, does not touch any widget logic */
    [data-testid="stExpander"],
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stCameraInput"] {
        background-color: rgba(255, 255, 255, 0.70);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.16);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(76, 29, 149, 0.08);
    }

    /* Soft card styling for Streamlit's built-in alert boxes
       (st.error/st.warning/st.info), which we already use for
       urgent flags and messages */
    [data-testid="stAlert"] {
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    }

    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div {
        border-radius: 12px !important;
    }

    audio {
        border-radius: 10px;
    }

    </style>
"""
st.markdown(BASE_THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIGN IN (demo login)
# ---------------------------------------------------------------------
# Simple fixed-credential gate — good enough for a demo/hackathon build.
# Change these via env vars / st.secrets (APP_USERNAME, APP_PASSWORD)
# before sharing the app with anyone else.
DEMO_USERNAME = get_key("APP_USERNAME") or "sahayak"
DEMO_PASSWORD = get_key("APP_PASSWORD") or "sahayak2026"


def check_login() -> bool:
    """Show a login form until the correct demo credentials are entered."""
    if st.session_state.get("authenticated"):
        return True

    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown(
            """
            <div style="text-align:center; margin: 48px 0 4px 0;">
                <h1 style="font-family:'Space Grotesk','Inter',sans-serif;
                           color:#3B0764; margin-bottom:2px;">🗣️ SAHAYAK</h1>
                <p style="color:#6B21A8; margin-top:0;">Sign in to continue</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                if hmac.compare_digest(username, DEMO_USERNAME) and hmac.compare_digest(password, DEMO_PASSWORD):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        st.caption(f"Demo login — username: `{DEMO_USERNAME}` · password: `{DEMO_PASSWORD}`")
    return False


if not check_login():
    st.stop()

# ------------------------- ACCESSIBILITY DISPLAY MODE -------------------------
with st.sidebar:
    st.header("⚙️ Display Settings")
    high_contrast = st.toggle("🔎 Large text / High contrast mode", key="high_contrast")

    st.markdown("---")
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("---")
    st.header("🕘 History")
    if st.session_state.get("history"):
        for entry in st.session_state.history:
            with st.expander(f"{entry['mode']} · {entry['time']} · {entry['language']}"):
                st.caption("Original")
                st.write(entry["original"])
                st.caption("Result")
                st.write(entry["result"])
    else:
        st.caption("Nothing processed yet. Your readings will show up here.")

if high_contrast:
    st.markdown("""
        <style>
        html, body, [class*="css"], .stApp, section[data-testid="stSidebar"] {
            font-size: 20px !important;
            background: #000000 !important;
            background-color: #000000 !important;
            color: #FFFF00 !important;
        }
        .stButton button {
            font-size: 20px !important;
            background-color: #FFFF00 !important;
            color: #000000 !important;
            font-weight: bold !important;
        }
        h1, h2, h3 { color: #FFFF00 !important; }
        h1::after { background: #FFFF00 !important; }
        </style>
    """, unsafe_allow_html=True)

banner_bg = "linear-gradient(90deg, #FFD600, #FFFF00)" if high_contrast else "linear-gradient(100deg, #7C3AED 0%, #6366F1 50%, #06B6D4 100%)"
banner_title_color = "#000000" if high_contrast else "white"
banner_sub_color = "#000000" if high_contrast else "#EDE9FE"

st.markdown(
    f"""
    <div style="background: {banner_bg};
                padding: 20px 26px; border-radius: 18px; margin-bottom: 16px;
                box-shadow: 0 10px 30px rgba(99, 102, 241, 0.28);">
        <h1 style="font-family: 'Space Grotesk','Inter',sans-serif; color: {banner_title_color} !important; margin: 0;">🗣️ SAHAYAK</h1>
        <p style="color: {banner_sub_color}; margin: 4px 0 0 0; font-size: 15px;">
            Accessibility Reader — for documents, signs, text, and conversations
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Document Reader", "⌨️ Text Message", "🪧 Sign & Menu Reader", "🎙️ Voice Conversation", "🚨 Emergency"]
)

# ------------------------- TAB 1: DOCUMENT READER -------------------------
with tab1:
    st.write("Upload a photo of a form, notice, or letter. We'll simplify it, "
             "flag anything urgent, translate it, and read it aloud.")

    doc_input_mode = st.radio("Input method:", ["📁 Upload a file", "📷 Take a photo"], key="doc_mode", horizontal=True)

    if doc_input_mode == "📁 Upload a file":
        doc_file = st.file_uploader("Upload a document photo or PDF", type=["jpg", "jpeg", "png", "pdf"], key="doc")
    else:
        doc_file = st.camera_input("Take a photo of the document", key="doc_cam")

    doc_lang = st.selectbox("Read aloud in:", list(LANGUAGES.keys()), key="doc_lang")

    if doc_file:
        image = load_image_from_upload(doc_file)
        st.image(image, caption="Document", use_container_width=True)

        if st.button("🔍 Process Document", key="doc_btn"):
            with st.spinner("Reading the document..."):
                raw_text, confidence = extract_text_from_image(image)

            if confidence < 40 or len(raw_text) < 5:
                st.session_state.doc_result = None
                st.warning(
                    "⚠️ The photo is too blurry or unclear to read reliably. "
                    "Please retake the photo with better lighting and try again."
                )
            else:
                with st.spinner("Simplifying and checking for urgent info..."):
                    result = simplify_and_analyze(raw_text)

                with st.spinner(f"Translating to {doc_lang}..."):
                    translated = translate_text(result["simplified_text"], doc_lang)
                    translated_urgent = ""
                    if result["has_urgent_info"] and doc_lang != "English":
                        translated_urgent = translate_text(result["urgent_summary"], doc_lang)

                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(translated, LANGUAGES[doc_lang])

                # Store everything needed to render results — this survives
                # later reruns (e.g. clicking the QR button) since it lives
                # in session_state, not just this button click's run.
                st.session_state.doc_result = {
                    "raw_text": raw_text,
                    "simplified_text": result["simplified_text"],
                    "has_urgent_info": result["has_urgent_info"],
                    "urgent_summary": result["urgent_summary"],
                    "translated": translated,
                    "translated_urgent": translated_urgent,
                    "doc_lang": doc_lang,
                    "audio_path": audio_path,
                }
                add_to_history("Document Reader", raw_text, result["simplified_text"], doc_lang)

    # Render from session_state — independent of which button was clicked,
    # so the QR button (and anything else here) keeps working on reruns.
    if st.session_state.get("doc_result"):
        r = st.session_state.doc_result

        with st.expander("Show raw extracted text"):
            st.write(r["raw_text"])

        st.subheader("✅ Simplified Explanation")
        st.write(r["simplified_text"])

        if r["has_urgent_info"]:
            st.error(f"⏰ Urgent: {r['urgent_summary']}")

        if r["doc_lang"] != "English":
            st.subheader(f"🌐 In {r['doc_lang']}")
            st.write(r["translated"])
            if r["has_urgent_info"]:
                st.error(f"⏰ {r['translated_urgent']}")

        st.subheader("🔊 Listen")
        st.audio(r["audio_path"])

        render_qr_share_button(r["translated"], key="doc_qr_btn")


# ------------------------- TAB 2: TEXT MESSAGE -------------------------
with tab2:
    st.write("Type or paste any text — a message, an email, a notice — and get it "
             "simplified, translated, and read aloud.")

    typed_text = st.text_area("Enter text here:", height=150, key="typed_text")
    text_lang = st.selectbox("Read aloud in:", list(LANGUAGES.keys()), key="text_lang")

    if st.button("✨ Process Text", key="text_btn"):
        if not typed_text.strip():
            st.warning("Please type or paste some text first.")
            st.session_state.text_result = None
        else:
            with st.spinner("Simplifying..."):
                result = simplify_and_analyze(typed_text)

            with st.spinner(f"Translating to {text_lang}..."):
                translated = translate_text(result["simplified_text"], text_lang)
                translated_urgent = ""
                if result["has_urgent_info"] and text_lang != "English":
                    translated_urgent = translate_text(result["urgent_summary"], text_lang)

            with st.spinner("Generating audio..."):
                audio_path = text_to_speech(translated, LANGUAGES[text_lang], "text_output.mp3")

            st.session_state.text_result = {
                "typed_text": typed_text,
                "simplified_text": result["simplified_text"],
                "has_urgent_info": result["has_urgent_info"],
                "urgent_summary": result["urgent_summary"],
                "translated": translated,
                "translated_urgent": translated_urgent,
                "text_lang": text_lang,
                "audio_path": audio_path,
            }
            add_to_history("Text Message", typed_text, result["simplified_text"], text_lang)

    if st.session_state.get("text_result"):
        r = st.session_state.text_result

        st.subheader("✅ Simplified Explanation")
        st.write(r["simplified_text"])

        if r["has_urgent_info"]:
            st.error(f"⏰ Urgent: {r['urgent_summary']}")

        if r["text_lang"] != "English":
            st.subheader(f"🌐 In {r['text_lang']}")
            st.write(r["translated"])
            if r["has_urgent_info"]:
                st.error(f"⏰ {r['translated_urgent']}")

        st.subheader("🔊 Listen")
        st.audio(r["audio_path"])

        render_qr_share_button(r["translated"], key="text_qr_btn")


# ------------------------- TAB 3: SIGN & MENU READER -------------------------
with tab3:
    st.write("Upload a photo of a street sign, menu, ticket board, or shop sign. "
             "We'll translate it into your language and read it aloud — useful "
             "when visiting a new place where signs aren't in your language.")

    sign_input_mode = st.radio("Input method:", ["📁 Upload a file", "📷 Take a photo"], key="sign_mode", horizontal=True)

    if sign_input_mode == "📁 Upload a file":
        sign_file = st.file_uploader("Upload a photo or PDF of a sign or menu", type=["jpg", "jpeg", "png", "pdf"], key="sign")
    else:
        sign_file = st.camera_input("Take a photo of the sign/menu", key="sign_cam")

    sign_lang = st.selectbox("Read aloud in:", list(LANGUAGES.keys()), key="sign_lang")

    if sign_file:
        image = load_image_from_upload(sign_file)
        st.image(image, caption="Sign/menu", use_container_width=True)

        if st.button("🪧 Translate Sign", key="sign_btn"):
            with st.spinner("Reading the sign..."):
                raw_text, confidence = extract_text_from_image(image)

            if confidence < 30 or len(raw_text) < 2:
                st.session_state.sign_result = None
                st.warning(
                    "⚠️ Couldn't read this clearly. Try getting closer or "
                    "reducing glare, then retake the photo."
                )
            else:
                with st.spinner(f"Translating to {sign_lang}..."):
                    translated = translate_text(raw_text, sign_lang)

                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(translated, LANGUAGES[sign_lang], "sign_output.mp3")

                st.session_state.sign_result = {
                    "raw_text": raw_text,
                    "translated": translated,
                    "sign_lang": sign_lang,
                    "audio_path": audio_path,
                }
                add_to_history("Sign & Menu Reader", raw_text, translated, sign_lang)

    if st.session_state.get("sign_result"):
        r = st.session_state.sign_result

        with st.expander("Show raw extracted text"):
            st.write(r["raw_text"])

        st.subheader(f"🌐 In {r['sign_lang']}")
        st.write(r["translated"])

        st.subheader("🔊 Listen")
        st.audio(r["audio_path"])

        render_qr_share_button(r["translated"], key="sign_qr_btn")


# ------------------------- TAB 4: VOICE CONVERSATION -------------------------
with tab4:
    st.write("**Live multi-person conversation.** Each person records their turn in "
             "their own language; SAHAYAK transcribes, detects the tone, translates "
             "it, and reads it aloud — building a running transcript everyone can follow.")

    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.caption(f"👥 {len(st.session_state.conversation_log)} turn(s) in this conversation")
    with col_b:
        if st.button("🗑️ Clear conversation", key="clear_convo"):
            st.session_state.conversation_log = []
            st.rerun()

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        speaker_name = st.text_input("Speaker's name:", value=f"Person {len(st.session_state.conversation_log) + 1}", key="speaker_name")
    with col2:
        spoken_lang = st.selectbox("Speaking in:", list(LANGUAGES.keys()), key="spoken_lang")
    with col3:
        reply_lang = st.selectbox("Translate to:", list(LANGUAGES.keys()), index=1, key="reply_lang")

    audio_input = st.audio_input("Record this turn", key="voice_input")

    if audio_input is not None:
        st.audio(audio_input)

        if st.button("🎙️ Add to conversation", key="voice_btn"):
            with st.spinner("Listening..."):
                try:
                    audio_bytes = audio_input.getvalue()
                    transcript = transcribe_audio(audio_bytes, SPEECH_LANGUAGES[spoken_lang])
                except sr.UnknownValueError:
                    st.error("Couldn't understand the audio. Please speak clearly and try again.")
                    transcript = None
                except sr.RequestError:
                    st.error("Speech recognition service is unavailable right now. Check your internet connection.")
                    transcript = None

            if transcript:
                with st.spinner(f"Detecting tone and translating to {reply_lang}..."):
                    result = detect_and_translate_tone(transcript, reply_lang, source_hint=spoken_lang)

                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(
                        result["translated_text"], LANGUAGES[reply_lang],
                        f"voice_output_{len(st.session_state.conversation_log)}.mp3"
                    )

                st.session_state.conversation_log.append({
                    "speaker": speaker_name,
                    "spoken_lang": spoken_lang,
                    "reply_lang": reply_lang,
                    "transcript": transcript,
                    "translated": result["translated_text"],
                    "tone_label": result["tone_label"],
                    "audio_path": audio_path,
                    "time": datetime.datetime.now().strftime("%I:%M %p"),
                })
                add_to_history("Voice Conversation", transcript, result["translated_text"], reply_lang)
                st.rerun()

    if st.session_state.conversation_log:
        st.markdown("### 💬 Conversation transcript")
        for turn in reversed(st.session_state.conversation_log):
            st.markdown(
                f"""
                <div style="background-color:#F5F5F5; border-left: 4px solid #1E88E5;
                            padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
                    <b>{turn['speaker']}</b> ({turn['spoken_lang']} → {turn['reply_lang']})
                    &nbsp;·&nbsp; {turn['tone_label']} &nbsp;·&nbsp;
                    <span style="color:#777; font-size: 12px;">{turn['time']}</span><br>
                    <span style="color:#555;">"{turn['transcript']}"</span><br>
                    <span style="font-weight: 600; color:#0D47A1;">→ {turn['translated']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.audio(turn["audio_path"])


# ------------------------- TAB 5: EMERGENCY MODE -------------------------
with tab5:
    st.error(
        "🚨 **Emergency phrases** — instant, pre-translated, works with weak or no "
        "internet for the text itself (audio playback still needs a connection)."
    )

    situation = st.selectbox("What's the emergency?", list(EMERGENCY_PHRASES.keys()), key="emergency_situation")
    emergency_lang = st.selectbox(
        "Show/speak this in:", list(LANGUAGES.keys()), key="emergency_lang"
    )

    phrase = EMERGENCY_PHRASES[situation][emergency_lang]

    st.markdown(
        f"""
        <div style="background-color:#FFEBEE; border-left: 6px solid #D32F2F;
                    padding: 20px; border-radius: 8px; margin-top: 12px;">
            <p style="font-size: 22px; font-weight: bold; color: #B71C1C; margin: 0;">
                {phrase}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔊 Speak this now", key="emergency_speak_btn"):
        with st.spinner("Generating audio..."):
            audio_path = text_to_speech(phrase, LANGUAGES[emergency_lang], "emergency_output.mp3")
        st.audio(audio_path, autoplay=True)
        add_to_history("Emergency", situation, phrase, emergency_lang)

    with st.expander("See this phrase in all languages"):
        for lang_name, text in EMERGENCY_PHRASES[situation].items():
            st.write(f"**{lang_name}:** {text}")


st.markdown("---")
st.caption("Built for Engineers' Day 2026 AI Agent Challenge · SAHAYAK")
