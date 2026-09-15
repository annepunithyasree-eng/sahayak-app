# 🗣️ SAHAYAK — Multilingual Accessibility Reader

An AI agent with five modes, built for people who need help reading,
understanding, and responding to text in daily life:

1. **📄 Document Reader** — photo of a form/notice → simplified, urgent-info
   flagged, translated, and read aloud
2. **⌨️ Text Message** — type or paste text → simplified, translated, read
   aloud
3. **🪧 Sign & Menu Reader** — photo of a sign/menu/board → translated, read
   aloud
4. **🎙️ Voice Conversation** — speak into the mic → transcribed, tone
   detected, translated, read aloud, built into a running transcript
5. **🚨 Emergency** — instant, pre-translated emergency phrases (fire,
   medical, police, accident, lost) — works with zero API calls

The app is gated behind a sign-in screen and supports a large-text /
high-contrast display mode for accessibility.

---

## 1. Install requirements

```bash
pip install -r requirements.txt
```

You'll also need Tesseract OCR installed on your machine (used for reading
text out of photos):

- **Streamlit Community Cloud**: already handled — it reads `packages.txt`
  automatically.
- **Local Mac**: `brew install tesseract`
- **Local Windows**: install from
  [UB-Mannheim's Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
- **Local Linux**: `sudo apt install tesseract-ocr`

## 2. Set up your Gemini API key

Get a free key at [aistudio.google.com](https://aistudio.google.com), then
set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"        # Mac/Linux
setx GEMINI_API_KEY "your-key-here"           # Windows
```

A second, optional backup key (`GEMINI_API_KEY_2`) can be set the same way —
the app automatically switches to it if the primary key's free-tier quota
runs out.

Free tier gives you about 1,500 requests/day — more than enough for building
and demoing.

### Sign-in

The app is protected by a simple demo login. Default credentials:

- **Username:** `sahayak`
- **Password:** `sahayak2026`

To change them, set `APP_USERNAME` / `APP_PASSWORD` the same way as the
Gemini keys above (env var locally, or `st.secrets` on Streamlit Cloud).
This is a fixed-credential gate meant for demos — not a real multi-user
account system.

## 3. Run the app

```bash
streamlit run app.py
```

It will open in your browser automatically (usually at
`http://localhost:8501`). Sign in with the credentials above to reach the
app.

## 4. Demo flow for judges

1. Sign in with the demo credentials
2. Upload a photo of a document (bank form, hospital notice, textbook page)
3. Click "Process Document"
4. Show the extracted text (expandable section)
5. Show the simplified explanation appear
6. Point out the urgent-info flag if the document has a deadline/amount
7. Switch the language dropdown to Telugu/Hindi/Tamil and re-translate
8. Play the audio out loud — this is the "wow" moment

## 5. Tips before demo day

- **Test on real, slightly messy photos** — not perfect scans. Judges will
  likely test with their own document.
- **Have a backup document ready** in case live WiFi/API is unreliable.
- **Practice explaining the "why not Google Lens" differentiator** — the
  agent simplifies + prioritizes urgency + self-corrects on bad photos,
  which plain OCR/translation tools don't do.
- If OCR confidence is too low, the app already asks the user to retake the
  photo automatically — show this behavior deliberately in the demo with a
  blurry test photo, it's a good "agentic" moment to highlight.

## Team

- Built by [Your Name] and [Partner's Name]
- Engineers' Day 2026 AI Agent Challenge — Accessibility Assistance Agent
