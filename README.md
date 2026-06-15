# Noema

**Understand your thoughts. Navigate your emotions. Move forward with clarity.**

Noema is an experimental psychology-informed AI companion built with Streamlit.
It helps users reflect on emotions, notice thinking patterns, make decisions,
and ask current factual or research-oriented questions through one chatbox.

Noema is not a therapist, psychologist, doctor, lawyer, or emergency service.

## Features

- Emotion Detection
- Cognitive Distortion Detection
- Decision Support
- Relationship Guidance
- Workplace Guidance
- Business Decision Support
- Research Retrieval
- Internet Search
- Safety Support
- Mixed Life Problem Analysis

## Architecture

```text
User Input
  ↓
Emotion Detection
  ↓
Intent Detection
  ↓
Topic Detection
  ↓
Strategy Selection
  ↓
Critic Layer
  ↓
Final Response
```

Noema routes each message before responding. Emotional messages stay grounded in
validation and exploration. Advice and decision requests receive practical
guidance. Current factual questions use Tavily when configured. Research
questions use Tavily with academic-domain filtering. Crisis language is handled
by a deterministic local safety layer before internet or AI calls.

## Screenshots

Screenshots will be added after public deployment.

Suggested screenshots:

- Main chat interface
- Response details panel
- Dashboard page
- Streamlit Cloud deployment page

## Datasets Used

Noema uses datasets for evaluation, routing examples, and few-shot behavior
guidance. It does not fine-tune a model in this repository.

- GoEmotions
- Empathetic Dialogues
- CounselChat
- HH-RLHF
- Noema Dataset
- Noema Decision Dataset

## Running Locally

Install Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Optional local secrets:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then add your own values:

```toml
TAVILY_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-5.5"
ENABLE_WEB_SEARCH = true
```

Tavily enables live factual search and academic research retrieval. OpenAI is
optional; when it is unavailable, Noema keeps local emotional support, decision
support, memory, internet search, and Tavily-based research retrieval working.

## Deployment

### Streamlit Cloud

1. Push this repository to GitHub.
2. Open Streamlit Cloud and choose **New app**.
3. Select the repository, branch, and `app.py` as the entry point.
4. Add secrets from `.streamlit/secrets.example.toml`.
5. Deploy.

### Hugging Face Spaces

1. Create a new Space.
2. Choose **Streamlit** as the SDK.
3. Upload or connect this repository.
4. Add secrets in Space settings.
5. Ensure `requirements.txt` is present.
6. Launch the Space.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full deployment guide.

## Research Vision

Noema is an experimental psychology-informed AI assistant. The long-term vision
is to study how routing, response style, memory, source attribution, and
structured reflection can improve emotionally intelligent AI systems without
turning every conversation into therapy mode.

The project is designed around transparent routing, lightweight local safety
rules, source-aware retrieval, and measurable evaluation before any future
fine-tuning.

## Privacy Notes

Noema stores anonymous labels and feedback for local analytics. It does not
store raw chat transcripts in the analytics database. Session memory is opt-in
and can be cleared from the sidebar.

If OpenAI is configured, recent conversation context may be sent to OpenAI for
enhanced response generation. Tavily receives the query only when the route
requires current facts or research retrieval.

## Tests

```bash
python -m pytest
python scripts/run_dataset_evaluation.py
python scripts/audit_safety_routing.py
```

## Disclaimer

Noema is not a therapist, psychologist, doctor, lawyer, or emergency service.
It does not diagnose, treat, or replace professional care. If someone may be in
immediate danger or unable to stay safe, they should contact local emergency
services or a crisis hotline immediately.
