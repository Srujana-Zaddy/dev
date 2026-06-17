"""
Truth Layer — a marketing-claim fact-checking agent.

Pipeline:
  PDF upload -> extract text -> Gemini extracts checkable claims (JSON)
  -> for each claim, Tavily searches the live web for evidence
  -> Gemini compares claim vs. evidence and classifies it
  -> Streamlit renders a color-coded results table.

Built for a zero-budget, free-tier-only stack:
  - Streamlit Community Cloud (hosting)
  - Gemini API (gemini-2.5-flash) for extraction + classification
  - Tavily Search API for live web evidence
"""

import io
import json
import os
import re
import time

import pandas as pd
import pdfplumber
import pypdf
import streamlit as st
from google import genai
from google.genai import types
from tavily import TavilyClient

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Truth Layer — Marketing Claim Fact-Checker",
    page_icon="🔍",
    layout="wide",
)

STATUS_COLORS = {
    "True": "#1a7f37",
    "False": "#d1242f",
    "Outdated": "#bf8700",
    "Unverifiable": "#6e7781",
}
STATUS_BG = {
    "True": "#dafbe1",
    "False": "#ffebe9",
    "Outdated": "#fff8c5",
    "Unverifiable": "#f0f1f3",
}

MAX_CLAIMS_DEFAULT = 12


def get_secret(key: str):
    """Read a secret from Streamlit's secrets manager, falling back to env vars
    so the same code works locally, on Streamlit Cloud, and in tests."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# --------------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF. Tries pdfplumber first (better layout
    handling for tables/sidebars common in marketing PDFs), falls back to
    pypdf if that fails or returns nothing."""
    text_parts = []
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception:
        text_parts = []

    text = "\n".join(text_parts).strip()

    if not text:
        try:
            uploaded_file.seek(0)
            reader = pypdf.PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts).strip()
        except Exception as e:
            raise RuntimeError(f"Could not extract text from this PDF: {e}")

    if not text:
        raise RuntimeError(
            "No extractable text found in this PDF. It may be a scanned "
            "image without an OCR layer."
        )
    return text


# --------------------------------------------------------------------------
# JSON helpers — Gemini sometimes wraps JSON in markdown fences despite
# instructions, so we defensively strip those before parsing.
# --------------------------------------------------------------------------

def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _call_gemini(client, model: str, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with light retry/backoff for transient rate-limit errors,
    since the free tier is the most likely failure point during live review."""
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Step 1 — claim extraction
# --------------------------------------------------------------------------

EXTRACT_PROMPT = """You are analyzing a marketing document to find factual claims \
that can be checked against real-world evidence.

Extract every checkable factual claim: statistics, dates, market figures, \
named comparisons to competitors, named historical or geographic facts. Skip \
pure marketing opinion with no checkable fact (e.g. "industry-leading", \
"best-in-class") unless it includes a specific number or comparison.

Return ONLY a JSON array, no markdown fences, no other text. Each item must \
have exactly this shape:
[{{"claim": "<the claim restated as one self-contained sentence>", \
"context": "<the surrounding sentence or two from the document, verbatim>"}}]

Limit yourself to the {max_claims} most checkable, highest-confidence claims.

Document:
\"\"\"
{document_text}
\"\"\"
"""


def extract_claims(client, model: str, document_text: str, max_claims: int) -> list:
    prompt = EXTRACT_PROMPT.format(
        max_claims=max_claims, document_text=document_text[:15000]
    )
    raw = _call_gemini(client, model, prompt)
    cleaned = _clean_json(raw)
    claims = json.loads(cleaned)
    if not isinstance(claims, list):
        raise ValueError("Expected a JSON array of claims from Gemini.")
    return claims[:max_claims]


# --------------------------------------------------------------------------
# Step 2 — live web evidence via Tavily
# --------------------------------------------------------------------------

def search_evidence(tavily_client, claim_text: str):
    try:
        return tavily_client.search(
            query=claim_text,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Step 3 — classification
# --------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are a fact-checker comparing a claim from a marketing \
document against live web search evidence.

Claim: "{claim}"
Original context: "{context}"

Web evidence:
{evidence}

Classify the claim as exactly one of: "True", "False", "Outdated", "Unverifiable".
- "Outdated": likely true at some point, but evidence shows a materially \
different current figure or status.
- "Unverifiable": no usable evidence either way, and the claim isn't \
implausible on its face.
- "False": evidence contradicts the claim, OR no evidence exists and the \
claim is specific/implausible enough to flag as unsupported.

Return ONLY a JSON object, no markdown fences, no other text, exactly:
{{"status": "<True|False|Outdated|Unverifiable>", "correct_info": "<if False \
or Outdated, the correct current fact in one sentence from the evidence; \
otherwise empty string>", "reasoning": "<one sentence>"}}
"""


def classify_claim(client, model: str, claim: str, context: str, evidence) -> dict:
    # No evidence at all is a known failure mode — handle it deterministically
    # instead of letting the model guess, per the "fail loud, don't crash"
    # design: flag it rather than silently skipping.
    if not evidence or not evidence.get("results"):
        return {
            "status": "False",
            "correct_info": "",
            "reasoning": "No web evidence could be found to support this claim.",
        }

    snippets = []
    if evidence.get("answer"):
        snippets.append(f"Search summary: {evidence['answer']}")
    for r in evidence.get("results", [])[:5]:
        title = r.get("title", "")
        content = (r.get("content", "") or "")[:500]
        url = r.get("url", "")
        snippets.append(f"- {title}: {content} (source: {url})")
    evidence_text = "\n".join(snippets)

    prompt = CLASSIFY_PROMPT.format(claim=claim, context=context, evidence=evidence_text)
    raw = _call_gemini(client, model, prompt)
    cleaned = _clean_json(raw)
    result = json.loads(cleaned)
    for key in ("status", "correct_info", "reasoning"):
        result.setdefault(key, "")
    if result["status"] not in STATUS_COLORS:
        result["status"] = "Unverifiable"
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(client, model, tavily_client, document_text, max_claims, progress_cb=None):
    claims = extract_claims(client, model, document_text, max_claims)
    results = []
    for i, c in enumerate(claims):
        claim_text = c.get("claim", "").strip()
        context = c.get("context", "").strip()
        if not claim_text:
            continue
        if progress_cb:
            progress_cb(i, len(claims), claim_text)
        evidence = search_evidence(tavily_client, claim_text)
        sources = []
        if evidence and evidence.get("results"):
            sources = [r.get("url", "") for r in evidence["results"][:3] if r.get("url")]
        verdict = classify_claim(client, model, claim_text, context, evidence)
        results.append(
            {
                "Claim": claim_text,
                "Context": context,
                "Status": verdict["status"],
                "Correct Info": verdict["correct_info"],
                "Reasoning": verdict["reasoning"],
                "Sources": sources,
            }
        )
    return results


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def render_badge(status: str) -> str:
    fg = STATUS_COLORS.get(status, "#333")
    bg = STATUS_BG.get(status, "#eee")
    return (
        f'<span style="background-color:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">{status}</span>'
    )


def main():
    st.title("🔍 Truth Layer")
    st.caption(
        "Upload a marketing PDF. This agent extracts its factual claims, "
        "checks each one against live web evidence, and flags what's false "
        "or outdated — with the current correct figure where possible."
    )

    gemini_key = get_secret("GEMINI_API_KEY")
    tavily_key = get_secret("TAVILY_API_KEY")

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Gemini model",
            ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="Switch to Flash-Lite if you hit free-tier rate limits.",
        )
        max_claims = st.slider(
            "Max claims to check", min_value=3, max_value=25, value=MAX_CLAIMS_DEFAULT,
            help="Caps API usage — each claim costs 1 search call + 1 Gemini call.",
        )
        st.divider()
        st.caption("API keys are read from Streamlit secrets, never hardcoded.")
        if not gemini_key:
            st.error("GEMINI_API_KEY not set in secrets.")
        if not tavily_key:
            st.error("TAVILY_API_KEY not set in secrets.")

    uploaded_file = st.file_uploader("Upload a marketing PDF", type=["pdf"])

    if not uploaded_file:
        st.info("Waiting for a PDF to check.")
        return

    if not gemini_key or not tavily_key:
        st.error(
            "Missing API key(s). Add GEMINI_API_KEY and TAVILY_API_KEY in "
            "this app's Settings → Secrets (Streamlit Cloud) or in "
            ".streamlit/secrets.toml (local)."
        )
        return

    if st.button("Run fact-check", type="primary"):
        try:
            with st.spinner("Reading PDF…"):
                document_text = extract_pdf_text(uploaded_file)
        except RuntimeError as e:
            st.error(str(e))
            return

        client = genai.Client(api_key=gemini_key)
        tavily_client = TavilyClient(api_key=tavily_key)

        progress_bar = st.progress(0.0, text="Extracting claims…")
        status_text = st.empty()

        def progress_cb(i, total, claim_text):
            frac = (i + 1) / max(total, 1)
            progress_bar.progress(frac, text=f"Checking claim {i + 1}/{total}")
            status_text.caption(claim_text[:120])

        try:
            with st.spinner("Extracting claims and verifying against the live web…"):
                results = run_pipeline(
                    client, model, tavily_client, document_text, max_claims, progress_cb
                )
        except json.JSONDecodeError:
            st.error(
                "Gemini returned malformed JSON for this document. Try again, "
                "or switch models in the sidebar."
            )
            return
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.warning("No checkable factual claims were found in this document.")
            return

        st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]

        counts = pd.Series([r["Status"] for r in results]).value_counts()
        cols = st.columns(4)
        for col, status in zip(cols, ["True", "False", "Outdated", "Unverifiable"]):
            col.metric(status, int(counts.get(status, 0)))

        st.divider()

        for r in results:
            with st.container(border=True):
                st.markdown(
                    f"{render_badge(r['Status'])}&nbsp;&nbsp;**{r['Claim']}**",
                    unsafe_allow_html=True,
                )
                if r["Status"] in ("False", "Outdated") and r["Correct Info"]:
                    st.markdown(f"✅ **Current correct info:** {r['Correct Info']}")
                st.caption(r["Reasoning"])
                with st.expander("Original context & sources"):
                    st.write(r["Context"])
                    for s in r["Sources"]:
                        st.markdown(f"- [{s}]({s})")

        df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "Sources"} for r in results]
        )
        st.download_button(
            "Download results as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="fact_check_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
