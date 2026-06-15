# Noema Deployment Guide

This guide prepares Noema for Streamlit Cloud, Hugging Face Spaces, and Render.

## Before Deployment

Run the local checks:

```bash
python -m pip install -r requirements.txt
python -m pytest
python scripts/run_dataset_evaluation.py
python scripts/audit_safety_routing.py
```

Confirm that no private secrets are committed:

```bash
git status
```

The file `.streamlit/secrets.toml` must stay local and must not be committed.

## A. Streamlit Cloud

1. Push the repository to GitHub.
2. Go to Streamlit Cloud.
3. Select **New app**.
4. Choose the GitHub repository and branch.
5. Set the main file path to:

```text
app.py
```

6. Open **Advanced settings** and add secrets:

```toml
TAVILY_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-5.5"
ENABLE_WEB_SEARCH = true
```

7. Deploy the app.
8. Open the app and test:

- Casual prompt
- Emotional prompt
- Decision-support prompt
- Current factual prompt
- Research prompt
- Crisis-resource prompt

## B. Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose **Streamlit** as the SDK.
3. Choose public or private visibility.
4. Upload this repository or connect it through Git.
5. Ensure the repository contains:

```text
app.py
requirements.txt
runtime.txt
```

6. Add secrets in **Settings -> Repository secrets**:

```text
TAVILY_API_KEY
OPENAI_API_KEY
OPENAI_MODEL
ENABLE_WEB_SEARCH
```

7. Restart the Space.
8. Test the same prompt set used for Streamlit Cloud.

## C. Render Deployment

Render can run Noema as a web service.

1. Push the repository to GitHub.
2. Create a new Render **Web Service**.
3. Connect the GitHub repository.
4. Use Python 3.11.
5. Set the build command:

```bash
pip install -r requirements.txt
```

6. Set the start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

7. Add environment variables:

```text
TAVILY_API_KEY
OPENAI_API_KEY
OPENAI_MODEL
ENABLE_WEB_SEARCH
```

8. Deploy and test the public URL.

## Production Notes

- Noema should launch with `streamlit run app.py`.
- Keep `.streamlit/secrets.toml` private.
- Do not commit local SQLite databases, uploads, cache files, or raw private data.
- Tavily is required for live internet and academic retrieval.
- OpenAI is optional for enhanced language generation.
- Noema is not a replacement for professional mental health care or emergency services.
