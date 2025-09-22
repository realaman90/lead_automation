"""Automation script using browser-use to find LinkedIn contacts for marketing agencies."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Set environment variables to disable telemetry before importing browser-use
os.environ['BROWSER_USE_DISABLE_TELEMETRY'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

# Import browser-use after setting environment variables
from browser_use import Agent, BrowserSession, ChatGoogle, ChatOpenAI
from browser_use.llm import models as llm_models

# Comprehensive suppression of aiohttp warnings
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
warnings.filterwarnings("ignore", message=".*unclosed.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Set up proper aiohttp logging to suppress warnings
logging.getLogger('aiohttp.client').setLevel(logging.ERROR)
logging.getLogger('aiohttp.connector').setLevel(logging.ERROR)

# Disable browser-use telemetry to prevent internal warnings
os.environ['BROWSER_USE_DISABLE_TELEMETRY'] = '1'

# Create a custom logging filter to suppress specific warnings
class UnclosedSessionFilter(logging.Filter):
    def filter(self, record):
        return not (
            'Unclosed client session' in record.getMessage() or
            'Unclosed connector' in record.getMessage() or
            'client_session' in record.getMessage()
        )

# Apply the filter to all loggers
logging.getLogger().addFilter(UnclosedSessionFilter())
logging.getLogger('browser_use').addFilter(UnclosedSessionFilter())
logging.getLogger('bubus').addFilter(UnclosedSessionFilter())

# Load environment variables so the Gemini API key is available before agent creation
load_dotenv()

DEFAULT_INPUT = Path("data/Leads - Marketing agencies.csv")
DEFAULT_OUTPUT = Path("data/Leads - Marketing agencies with LinkedIn.csv")
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CHROME_EXECUTABLE = Path(
    os.getenv("CHROME_EXECUTABLE", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
).expanduser()
DEFAULT_CHROME_USER_DATA_DIR = Path(
    os.getenv("CHROME_USER_DATA_DIR", "~/Library/Application Support/Google/Chrome")
).expanduser()

LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")

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


def create_browser(
    headless: bool = False,
    profile: str = "Profile 16",
    use_existing: bool = False,
) -> BrowserSession:
    """Create a BrowserSession tied to the user's Chrome profile."""

    if use_existing:
        return BrowserSession(
            cdp_url="http://localhost:9222",
            is_local=True,
            headless=headless,
            profile_directory=profile,
            minimum_wait_page_load_time=3.0,
            wait_for_network_idle_page_load_time=8.0,
            wait_between_actions=2.0,
        )

    if not DEFAULT_CHROME_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Chrome executable not found at {DEFAULT_CHROME_EXECUTABLE}. Set CHROME_EXECUTABLE environment variable."
        )

    if not DEFAULT_CHROME_USER_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Chrome user data directory not found at {DEFAULT_CHROME_USER_DATA_DIR}."
            " Set CHROME_USER_DATA_DIR environment variable."
        )

    # Use a temporary profile directory to avoid conflicts with existing Chrome instances
    import tempfile
    temp_profile_dir = tempfile.mkdtemp(prefix="browser_use_profile_")

    return BrowserSession(
        executable_path=str(DEFAULT_CHROME_EXECUTABLE),
        headless=headless,
        user_data_dir=temp_profile_dir,  # Use temporary profile to avoid conflicts
        profile_directory=None,  # Don't use specific profile in temp directory
        minimum_wait_page_load_time=3.0,
        wait_for_network_idle_page_load_time=8.0,
        wait_between_actions=2.0,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--no-sandbox",  # Required for some macOS setups
            "--disable-dev-shm-usage",  # Prevent /dev/shm issues
            "--disable-web-security",  # Allow cross-origin requests
            "--disable-features=VizDisplayCompositor",  # Fix rendering issues
            "--remote-debugging-port=0",  # Let Chrome choose a free port
            "--disable-extensions",  # Disable extensions for stability
            "--disable-plugins",  # Disable plugins for stability
            "--disable-images",  # Faster loading
            "--disable-javascript",  # Disable JS for faster loading (if needed)
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ],
    )

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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for failed agent runs",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (not recommended for LinkedIn)",
    )
    parser.add_argument(
        "--profile",
        default="Profile 16",
        help="Chrome profile to use (default: Profile 16)",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Connect to already running Chrome instance on port 9222",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def cleanup_aiohttp_sessions():
    """Clean up any remaining aiohttp sessions to prevent warnings."""
    try:
        import aiohttp
        import gc
        
        # Force cleanup of any remaining aiohttp sessions
        for obj in gc.get_objects():
            if isinstance(obj, aiohttp.ClientSession):
                try:
                    if not obj.closed:
                        asyncio.run(obj.close())
                except Exception:
                    pass
            elif isinstance(obj, aiohttp.TCPConnector):
                try:
                    if not obj.closed:
                        asyncio.run(obj.close())
                except Exception:
                    pass
        
        # Force garbage collection
        gc.collect()
    except Exception as e:
        logging.debug(f"Cleanup warning: {e}")


class BrowserSessionManager:
    """Context manager for proper browser session cleanup."""
    
    def __init__(self, browser_factory):
        self.browser_factory = browser_factory
        self.browser_session = None
        
    def __enter__(self):
        self.browser_session = self.browser_factory()
        return self.browser_session
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser_session:
            try:
                asyncio.run(self.browser_session.kill())
            except Exception as e:
                logging.debug(f"Browser session cleanup warning: {e}")
            finally:
                cleanup_aiohttp_sessions()


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


def build_task(
    company: str,
    location: str,
    website: str,
    login_instructions: Optional[str] = None,
) -> str:
    safe_company = company or "Unknown"
    safe_location = location or "Unknown"
    safe_website = website or "Unknown"
    task = TASK_TEMPLATE.format(company=safe_company, location=safe_location, website=safe_website)
    if login_instructions:
        task += "\n\nLinkedIn access notes:\n" + login_instructions
    return task


def run_agent_with_retry(
    task: str,
    model_name: str,
    max_steps: int,
    max_retries: int = 3,
    sensitive_data: Optional[dict[str, str]] = None,
    browser_factory=None,
) -> tuple[LinkedInContact | None, str | None]:
    """Run agent with retry logic for handling browser timeouts and errors."""
    for attempt in range(max_retries):
        browser_session = None
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} for agent task")

            # Try different browser strategies based on attempt number
            if attempt == 0 and browser_factory:
                # First attempt: use provided browser factory
                browser_session = browser_factory()
            elif attempt == 1:
                # Second attempt: try using existing Chrome instance
                logging.info("Trying to connect to existing Chrome instance...")
                browser_session = create_browser(use_existing=True)
            else:
                # Third attempt: launch new Chrome with minimal config
                logging.info("Launching new Chrome instance with minimal config...")
                browser_session = create_browser(headless=True)

            agent_kwargs = {
                "task": task,
                "llm": resolve_llm(model_name),
                "output_model_schema": LinkedInContact,
                "browser_session": browser_session,
                "disable_telemetry": True,  # Disable telemetry to reduce warnings
            }
            if sensitive_data:
                agent_kwargs["sensitive_data"] = sensitive_data

            # Suppress stderr during agent run to catch any remaining warnings
            import contextlib
            import sys

            async def run_once() -> tuple[LinkedInContact | None, str | None]:
                agent = None
                try:
                    with contextlib.redirect_stderr(open(os.devnull, 'w')):
                        agent = Agent(**agent_kwargs)
                        history = await agent.run(max_steps=max_steps)
                        final_text_inner = history.final_result()
                        try:
                            structured_inner = history.structured_output
                        except ValidationError as validation_error:
                            logging.warning("Structured output validation failed: %s", validation_error)
                            structured_inner = None
                        return structured_inner, final_text_inner
                finally:
                    if agent:
                        try:
                            await agent.close()
                        except Exception as close_error:
                            logging.warning(f"Error closing agent: {close_error}")
                        # Additional cleanup for agent's internal sessions
                        try:
                            if hasattr(agent, 'browser_session') and agent.browser_session:
                                await agent.browser_session.kill()
                        except Exception as browser_error:
                            logging.debug(f"Agent browser session cleanup warning: {browser_error}")

            structured, final_text = asyncio.run(run_once())
            return structured, final_text
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} attempts failed for task")
                return None, f"Agent failed after {max_retries} attempts: {e}"
        finally:
            # Ensure browser session is properly closed
            if browser_session:
                try:
                    # Force cleanup of all connections
                    asyncio.run(browser_session.kill())
                except Exception as kill_error:
                    logging.debug("Browser session cleanup warning: %s", kill_error)
                finally:
                    # Additional cleanup to prevent aiohttp warnings
                    try:
                        # Clean up any internal aiohttp sessions
                        if hasattr(browser_session, '_session') and browser_session._session:
                            asyncio.run(browser_session._session.close())
                        if hasattr(browser_session, '_cdp_session') and browser_session._cdp_session:
                            asyncio.run(browser_session._cdp_session.close())
                        if hasattr(browser_session, 'session') and browser_session.session:
                            asyncio.run(browser_session.session.close())
                        # Clean up any connector pools
                        if hasattr(browser_session, '_connector') and browser_session._connector:
                            asyncio.run(browser_session._connector.close())
                    except Exception as session_error:
                        logging.debug("Session cleanup warning: %s", session_error)
                    
                    # Force garbage collection to clean up any remaining references
                    import gc
                    gc.collect()
    
    return None, "Unexpected error in retry logic"


def run_agent(
    task: str,
    model_name: str,
    max_steps: int,
    sensitive_data: Optional[dict[str, str]] = None,
    browser_factory=None,
) -> tuple[LinkedInContact | None, str | None]:
    """Legacy function for backward compatibility."""
    return run_agent_with_retry(
        task,
        model_name,
        max_steps,
        sensitive_data=sensitive_data,
        browser_factory=browser_factory,
    )


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

    def browser_factory():
        return create_browser(headless=args.headless, profile=args.profile, use_existing=args.use_existing)

    login_instructions: Optional[str] = None
    sensitive_data: Optional[dict[str, str]] = None
    if LINKEDIN_EMAIL and LINKEDIN_PASSWORD:
        login_instructions = (
            "If LinkedIn prompts for authentication, sign in using these credentials directly in the form. "
            "Do not expose them in any notes or outputs.\n"
            f"Email: {LINKEDIN_EMAIL}\n"
            f"Password: {LINKEDIN_PASSWORD}"
        )
        sensitive_data = {
            "linkedin_email": LINKEDIN_EMAIL,
            "linkedin_password": LINKEDIN_PASSWORD,
        }
    elif LINKEDIN_EMAIL or LINKEDIN_PASSWORD:
        logging.warning(
            "LinkedIn credentials are partially configured. Set both LINKEDIN_EMAIL and LINKEDIN_PASSWORD in the environment to enable automatic sign-in."
        )

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
                structured, final_text = run_agent_with_retry(
                    build_task(company, location, website, login_instructions=login_instructions),
                    model_name=args.model,
                    max_steps=args.max_steps,
                    max_retries=args.max_retries,
                    sensitive_data=sensitive_data,
                    browser_factory=browser_factory,
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
    
    # Clean up any remaining aiohttp sessions
    cleanup_aiohttp_sessions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
