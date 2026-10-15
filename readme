# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get an
ATS score out of 100 plus prioritised improvements, keyword gaps and line rewrites.

**Stack:** Streamlit (UI) · Google Gemini `gemini-3.8-flash` (AI) · pypdf / python-docx (parsing)

## Features
- ATS score computed from 5 sub-scores: formatting, content quality, keywords, impact/metrics, structure
- Optional job description for an accurate keyword match
- Strengths, prioritised improvements, matched/missing keywords, before/after rewrites
- Scanned PDFs are sent to Gemini directly when no text can be extracted
- Structured JSON output, retries on transient errors

## Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- paste it into the app sidebar, **or**
- set it as an environment variable: `export GEMINI_API_KEY="your-key"` (Windows: `set GEMINI_API_KEY=your-key`), **or**
- create `.streamlit/secrets.toml` (never commit this file):
  ```toml
  GEMINI_API_KEY = "your-key"
  ```

```bash
streamlit run app.py
```

## Configuration
| Name | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API key | none (sidebar prompt) |
| `GEMINI_MODEL` | Model to use | `gemini-3.8-flash` |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public, or private with access granted).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Notes
- The score is an AI estimate, not the output of a real ATS. Treat it as guidance.
- Always check rewrites are truthful before using them.
- Resumes are sent to the Gemini API; check Google's data policy before uploading sensitive data.
- Gemini 3.8 Flash uses introductory pricing until 31 Dec 2026; check the pricing page.

## Files
- `app.py` – the app
- `requirements.txt` – dependencies
- `README.md` – this file
