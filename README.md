# Agencies Automation

Python helper script that uses the [browser-use](https://docs.browser-use.com/introduction) agent to search for high-value LinkedIn contacts for each marketing agency in `data/Leads - Marketing agencies.csv`.

## Prerequisites

- Python 3.11+
- Latest Google Gemini API key stored in a local `.env` file:
  ```env
  GEMINI_API_KEY=your-key-here
  ```
- Dependencies installed:
  ```bash
  pip install -r requirements.txt
  playwright install chromium --with-deps --no-shell
  ```
  > `browser-use` relies on Playwright with a Chromium build. If you already have one, you can skip the second command.

## Usage

```bash
python app.py \
  --input "data/Leads - Marketing agencies.csv" \
  --output "data/Leads - Marketing agencies with LinkedIn.csv" \
  --limit 5
```

Useful flags:
- `--limit N` – process only the first `N` unprocessed leads (handy for testing).
- `--skip N` – skip the first `N` unprocessed leads before starting.
- `--resume` – append to an existing output file and skip rows that already have results.
- `--model <name>` – choose a different LLM, e.g. `gemini-1.5-pro`.
- `--max-steps <int>` – cap the number of agent steps per company (default 30).

Results are stored (or appended) in `data/Leads - Marketing agencies with LinkedIn.csv` with extra columns for the contact, LinkedIn URL, confidence, and notes. Each run flushes results as soon as they are produced so you can safely interrupt and resume later.

## Notes

- The automation opens real browser sessions; runs can take several minutes depending on the number of leads.
- Keep your API keys secret—avoid committing `.env` or sharing command logs with sensitive data.
- Output rows include the raw JSON returned by the agent to help with auditing or manual corrections.
