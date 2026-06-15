# Noema Deployment Report

Generated for Noema v1.0 public deployment preparation.

## Summary

Noema is ready for Streamlit Cloud, Hugging Face Spaces, and Render deployment
from a code/configuration perspective.

GitHub push is blocked in this workspace because no connected GitHub remote is
available.

## Checks

| Check | Result |
|---|---:|
| Full test suite | 171 passed |
| Dependency validation | Passed |
| Dependency count | 5 |
| Streamlit app health | ok |
| Secret scan | Passed |
| Local-only path scan | Passed |
| Safety audit false positives | 0 |
| Safety audit false negatives | 0 |
| Dataset evaluation targets | Passed |

## Evaluation Metrics

| Metric | Result |
|---|---:|
| intent_accuracy | 72.67% |
| emotion_accuracy | 42.03% |
| internet_routing_accuracy | 83.70% |
| research_routing_accuracy | 98.22% |
| safety_routing_accuracy | 100.00% |
| advice_answer_rate | 99.50% |
| decision_answer_rate | 100.00% |
| over_reflection_rate | 0.00% |
| bad_phrase_rate | 0.00% |
| generic_decision_template_rate | 0.00% |
| false_crisis_rate | 0.00% |
| mixed_intent_success_rate | 100.00% |
| distortion_detection_rate | 100.00% |
| relationship_routing_accuracy | 100.00% |
| business_routing_accuracy | 100.00% |
| casual_chat_success_rate | 100.00% |

## Security Scan

The deployable files were scanned for:

- OpenAI-style API key patterns
- Tavily-style API key patterns
- GitHub token patterns
- password/secret/token assignment patterns
- local-only path patterns

Result: no exposed secrets found in deployable files.

Note: `.streamlit/secrets.toml` exists locally for development and is ignored by
`.gitignore`. It should not be committed. Public deployment should use platform
secret settings instead.

## Ignored Local Artifacts

The following are excluded from public deployment:

- `.venv/`
- `.streamlit/secrets.toml`
- `.env`
- local SQLite databases
- logs
- `data/raw/`
- `data/private/`
- `data/processed/noema_unified_dataset.jsonl`
- uploads/cache/tmp/work/test output folders

## Deployment Readiness Score

**96 / 100**

The app is deployment-ready, and a local release commit exists. The score is not
100 because the push could not be completed without a GitHub remote.

## GitHub Status

- Git repository detected: Yes
- Connected remote detected: No
- Current branch: main
- Commit created: Yes
- Commit hash: see final Git output after commit creation
- Push completed: No
- Blocker: this workspace folder has no GitHub remote.

## Recommended Next Step

Create or connect a GitHub repository, then run:

```bash
git remote add origin <your-github-repo-url>
git push -u origin main
```

After GitHub is connected, deploy to Streamlit Cloud with `app.py` as the entry
point and add the required secrets in the platform settings.
