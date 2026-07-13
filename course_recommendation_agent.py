#!/usr/bin/env python3
"""
Course Recommendation Agent
============================

An agentic system that reads a student's profile (background, goal,
known skills) and a course catalogue (with prerequisites), then uses an
LLM to REASON about which courses that student should take, in what
order, and WHY -- producing a personalised learning path.

Nothing about the recommendation logic is hardcoded: the agent does not
use fixed if/else rules to pick courses. The LLM is given the full
catalogue and the student profile and must reason about prerequisite
chains, skill gaps, and goal alignment itself, then justify each step.

Usage
-----
    python course_recommendation_agent.py                     # run all sample profiles
    python course_recommendation_agent.py --profile "Aisha"   # run one profile
    python course_recommendation_agent.py --catalogue my_catalogue.json --profiles my_profiles.json

Setup
-----
    pip install -r requirements.txt
    export GROQ_API_KEY="gsk_..."       # free key: console.groq.com/keys
"""

import os
import sys
import json
import argparse
from pathlib import Path

import requests

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()  # groq | openai | anthropic | gemini
_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.0-flash",
}
MODEL_NAME = os.environ.get("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "llama-3.3-70b-versatile"))
OUTPUT_DIR = Path(__file__).parent / "outputs"


def call_llm(prompt: str, max_tokens: int = 1200) -> str:
    """Send a prompt to whichever LLM provider is configured (via
    LLM_PROVIDER env var) and return the raw text response.

    Implemented with plain REST calls (requests) rather than each
    provider's SDK, so no extra packages are needed to switch providers
    -- just an API key and LLM_PROVIDER=<name>.

    Supported: groq (default, free), openai, anthropic, gemini.
    """
    if LLM_PROVIDER == "groq":
        return _call_openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions",
            "GROQ_API_KEY", prompt, max_tokens,
            key_hint="https://console.groq.com/keys (free, no card)",
        )

    elif LLM_PROVIDER == "openai":
        return _call_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            "OPENAI_API_KEY", prompt, max_tokens,
            key_hint="https://platform.openai.com/api-keys",
        )

    elif LLM_PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    elif LLM_PROVIDER == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set the GEMINI_API_KEY environment variable first.")
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent",
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    else:
        sys.exit(f"ERROR: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
                  f"Use one of: groq, openai, anthropic, gemini.")


def _call_openai_compatible(url: str, key_var: str, prompt: str, max_tokens: int, key_hint: str) -> str:
    """Groq and OpenAI both speak the same chat-completions API shape."""
    api_key = os.environ.get(key_var)
    if not api_key:
        sys.exit(f"ERROR: Set the {key_var} environment variable first ({key_hint}).")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def build_prompt(catalogue: list, profile: dict) -> str:
    return f"""You are an autonomous course recommendation agent for an online
learning platform.

COURSE CATALOGUE (each course has an id, name, skills it teaches,
prerequisite course ids, level, and duration in weeks):
{json.dumps(catalogue, indent=2)}

STUDENT PROFILE:
{json.dumps(profile, indent=2)}

Your task: reason step by step about this student's current skill gaps
relative to their stated goal, then recommend an ORDERED learning path
of courses from the catalogue above that gets them there.

Rules you must apply through your own reasoning (do not just pattern
match on course names):
- Never recommend a course before its prerequisites are satisfied
  (either already known by the student, or earlier in the same path).
- Do not recommend courses irrelevant to the student's stated goal.
- Prefer the shortest sensible path -- do not pad with unnecessary courses.
- Every course you include must have a specific reason tied to this
  student's background or goal, not a generic justification.

Respond with ONLY valid JSON in this exact shape, nothing else:
{{
  "student": "<name>",
  "goal": "<their goal>",
  "skill_gap_analysis": "<1-3 sentences on what they're missing>",
  "recommended_path": [
    {{
      "step": 1,
      "course_id": "...",
      "course_name": "...",
      "reason": "specific reason tied to this student"
    }}
  ],
  "estimated_total_weeks": <int>
}}"""


def recommend_for_profile(catalogue: list, profile: dict) -> dict:
    prompt = build_prompt(catalogue, profile)
    text = call_llm(prompt)
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "student": profile.get("name", "unknown"),
            "goal": profile.get("goal", ""),
            "skill_gap_analysis": "LLM did not return valid JSON.",
            "recommended_path": [],
            "estimated_total_weeks": 0,
            "raw_response": text,
        }


def result_to_markdown(result: dict) -> str:
    lines = [f"# Learning Path for {result.get('student', 'Student')}\n"]
    lines.append(f"**Goal:** {result.get('goal', '')}\n")
    lines.append(f"**Skill gap analysis:** {result.get('skill_gap_analysis', '')}\n")
    lines.append("## Recommended Path\n")
    for step in result.get("recommended_path", []):
        lines.append(
            f"{step.get('step')}. **{step.get('course_name')}** "
            f"(`{step.get('course_id')}`)\n   - Reason: {step.get('reason')}"
        )
    lines.append(f"\n**Estimated total duration:** {result.get('estimated_total_weeks', '?')} weeks")
    return "\n".join(lines)


def run(catalogue_path: str, profiles_path: str, only_profile: str | None):
    catalogue = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))

    if only_profile:
        profiles = [p for p in profiles if p["name"].lower() == only_profile.lower()]
        if not profiles:
            sys.exit(f"No profile named '{only_profile}' found in {profiles_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = []

    for profile in profiles:
        print(f"Reasoning about a learning path for {profile['name']}...")
        result = recommend_for_profile(catalogue, profile)
        all_results.append(result)

        md_path = OUTPUT_DIR / f"{profile['name'].lower()}_path.md"
        md_path.write_text(result_to_markdown(result), encoding="utf-8")
        print(f"  -> {md_path}")

    combined_path = OUTPUT_DIR / "all_recommendations.json"
    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nDone. Combined JSON: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Course Recommendation Agent")
    parser.add_argument("--catalogue", default=str(Path(__file__).parent / "catalogue.json"))
    parser.add_argument("--profiles", default=str(Path(__file__).parent / "student_profiles.json"))
    parser.add_argument("--profile", default=None, help="Run only this student's profile by name")
    args = parser.parse_args()
    run(args.catalogue, args.profiles, args.profile)


if __name__ == "__main__":
    main()
