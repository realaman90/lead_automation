"""Automation script using browser-use to find LinkedIn contacts for marketing agencies."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from browser_use import Agent, ChatGoogle, ChatOpenAI
from browser_use.llm import models as llm_models

# Load environment variables so the Gemini API key is available before agent creation
load_dotenv()

DEFAULT_INPUT = Path("data/Leads - Marketing agencies.csv")
DEFAULT_OUTPUT = Path("data/Leads - Marketing agencies with LinkedIn.csv")
DEFAULT_MODEL = "gpt-4.1-mini"

TASK_TEMPLATE = """
You are a meticulous sales researcher. Your goal is to surface the strongest outreach contact for the company below while first harvesting concrete contact details from the company's own website.

Company: {company}
Location: {location}
Website: {website}

Research workflow:
1. Start on the official website. Locate About, Team, Leadership, or Contact pages. Capture every usable outreach detail you find (names, roles, emails, phone numbers, contact forms) and list them in `website_contacts`.
2. From those website findings, pick the single best person for marketing, partnerships, or business development. Prioritise founders, CEOs, managing directors, or senior marketing leads that clearly serve this agency and location.
3. Use Google (and other public web results) to confirm that person's details and uncover additional evidence. Favour searches like "<Name>" "<Company>" and "<Company> leadership".
4. Only open LinkedIn after verifying the person via Google. If a canonical LinkedIn profile exists, capture its URL. If none is available, leave `linkedin_url` null and explain why.
5. Summarise why this person is the best outreach target, referencing the specific pages you used.

Rules:
- Record every distinct website-sourced contact detail you find; do not guess or infer email patterns.
- Respect captcha/rate limits. If a property blocks you, document it in `notes` and continue with other sources.
- Prefer official/company-controlled sources before third-party directories.
- Finish by calling the done action exactly once with JSON that matches this schema:
  {{
    "company": "{company}",
    "location": "{location}",
    "website": "{website}",
    "website_contacts": "<names/titles/emails/phones from the company site or null>",
    "contact_name": "<best outreach person or null>",
    "title": "<job title or null>",
    "contact_email": "<direct email or null>",
    "contact_phone": "<direct phone or null>",
    "linkedin_url": "<https://... or null>",
    "source_urls": "<comma-separated URLs checked>",
    "confidence": <integer 0-10 representing confidence>,
    "notes": "<short explanation with references to the pages you used>"
  }}
- Do not add Markdown, explanations, or any text outside the JSON when calling done.
""".strip()


class LinkedInContact(BaseModel):
    company: str
    location: str | None = None
    website: str | None = None
    website_contacts: str | None = Field(
        None, description="Details gathered directly from the company's website"
    )
    contact_name: str | None = Field(None, description="Full name of the suggested contact")
    title: str | None = Field(None, description="Current role of the suggested contact")
    contact_email: str | None = Field(None, description="Direct email for the suggested contact")
    contact_phone: str | None = Field(None, description="Direct phone number for the suggested contact")
    linkedin_url: str | None = Field(None, description="LinkedIn profile URL starting with https://")
    source_urls: str | None = Field(
        None, description="Comma-separated list of URLs consulted during research"
    )
    confidence: int = Field(0, ge=0, le=10, description="Confidence score from 0 (low) to 10 (high)")
    notes: str | None = Field(None, description="Summary of why this contact was chosen and citations")


def resolve_llm(model_name: str):
    """Return a browser-use chat model instance based on user configuration."""

    candidate = (model_name or "").strip()
    if not candidate:
        raise ValueError("Model name cannot be empty.")

    try:
        return llm_models.get_llm_by_name(candidate)
    except ValueError:
        normalized = candidate.lower()
        if normalized.startswith("gemini") or "gemini" in normalized or normalized.startswith("google"):
            return ChatGoogle(model=candidate)
        return ChatOpenAI(model=candidate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find LinkedIn contacts for each marketing agency using browser-use.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the CSV file containing agencies (default: data/Leads - Marketing agencies.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the augmented CSV with LinkedIn data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of leads processed (useful for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the existing output file and skip leads that have already been processed",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model to use with browser-use (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of agent steps per lead",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N leads (applies after any resume filtering)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_leads(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        leads = [row for row in reader if any(row.values())]
    return leads


def load_processed_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    with output_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return {(row.get("Agency Name", ""), row.get("Location", "")) for row in reader}


def ensure_output_schema(output_path: Path, fieldnames: list[str]) -> None:
    """Align existing output files with the current schema when resuming runs."""

    if not output_path.exists():
        return

    with output_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []

    if not existing_fields or existing_fields == fieldnames:
        return

    logging.info("Updating %s to the latest output schema", output_path)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_task(company: str, location: str, website: str) -> str:
    safe_company = company or "Unknown"
    safe_location = location or "Unknown"
    safe_website = website or "Unknown"
    return TASK_TEMPLATE.format(company=safe_company, location=safe_location, website=safe_website)


def run_agent(task: str, model_name: str, max_steps: int) -> tuple[LinkedInContact | None, str | None]:
    agent = Agent(
        task=task,
        llm=resolve_llm(model_name),
        output_model_schema=LinkedInContact,
    )
    history = agent.run_sync(max_steps=max_steps)
    final_text = history.final_result()
    try:
        structured = history.structured_output
    except ValidationError as validation_error:
        logging.warning("Structured output validation failed: %s", validation_error)
        structured = None
    return structured, final_text


def prepare_row(
    lead: dict[str, str],
    structured: LinkedInContact | None,
    final_text: str | None,
) -> dict[str, str]:
    base_row = {
        "Agency Name": lead.get("Agency Name", ""),
        "Location": lead.get("Location", ""),
        "Website": lead.get("Website", ""),
    }

    if structured is not None:
        contact = structured.model_dump()
        base_row.update(
            {
                "website_contacts": contact.get("website_contacts") or "",
                "contact_name": contact.get("contact_name") or "",
                "title": contact.get("title") or "",
                "contact_email": contact.get("contact_email") or "",
                "contact_phone": contact.get("contact_phone") or "",
                "linkedin_url": contact.get("linkedin_url") or "",
                "source_urls": contact.get("source_urls") or "",
                "confidence": str(contact.get("confidence") if contact.get("confidence") is not None else ""),
                "notes": contact.get("notes") or "",
                "raw_result": final_text or "",
            }
        )
        return base_row

    logging.warning("No structured output produced; storing raw result for review.")
    raw_payload = final_text or ""
    if final_text:
        try:
            parsed = json.loads(final_text)
            if isinstance(parsed, dict):
                base_row.update({k: str(v) for k, v in parsed.items()})
        except json.JSONDecodeError:
            logging.debug("Agent output was not valid JSON; retaining raw text.")
    base_row.update(
        {
            "website_contacts": "",
            "contact_name": "",
            "title": "",
            "contact_email": "",
            "contact_phone": "",
            "linkedin_url": "",
            "source_urls": "",
            "confidence": "",
            "notes": "Agent did not return structured data",
            "raw_result": raw_payload,
        }
    )
    return base_row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    leads = load_leads(args.input)
    logging.info("Loaded %d leads", len(leads))

    processed_keys = load_processed_keys(args.output) if args.resume else set()
    if processed_keys:
        logging.info("Resuming run; skipping %d already processed leads", len(processed_keys))

    fieldnames = [
        "Agency Name",
        "Location",
        "Website",
        "website_contacts",
        "contact_name",
        "title",
        "contact_email",
        "contact_phone",
        "linkedin_url",
        "source_urls",
        "confidence",
        "notes",
        "raw_result",
    ]

    if args.resume and args.output.exists():
        ensure_output_schema(args.output, fieldnames)

    mode = "a" if args.resume and args.output.exists() else "w"
    total_targets = sum(
        1
        for lead in leads
        if (lead.get("Agency Name", ""), lead.get("Location", "")) not in processed_keys
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        processed_count = 0
        skipped_after_filter = 0

        for index, lead in enumerate(leads):
            key = (lead.get("Agency Name", ""), lead.get("Location", ""))
            if key in processed_keys:
                continue
            if skipped_after_filter < args.skip:
                skipped_after_filter += 1
                continue
            if args.limit is not None and processed_count >= args.limit:
                break

            company = lead.get("Agency Name", "")
            location = lead.get("Location", "")
            website = lead.get("Website", "")

            logging.info(
                "[%d/%d] Processing %s",
                processed_count + skipped_after_filter + 1,
                total_targets if args.limit is None else min(args.limit, total_targets),
                company or "Unnamed agency",
            )
            try:
                structured, final_text = run_agent(
                    build_task(company, location, website),
                    model_name=args.model,
                    max_steps=args.max_steps,
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("Agent failed for %s: %s", company, exc)
                structured = None
                final_text = f"Agent run failed: {exc}"

            row = prepare_row(lead, structured, final_text)
            processed_count += 1

            writer.writerow(row)
            csvfile.flush()

    if processed_count == 0:
        logging.info("No new leads processed; nothing to write.")
        return 0

    logging.info("Completed processing %d leads. Results stored in %s", processed_count, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
