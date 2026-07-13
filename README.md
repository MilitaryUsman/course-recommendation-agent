# Course Recommendation Agent

Built for the **Rooman Technologies 24-Hour AI Agent Challenge** —
Category 1: **Course Recommendation Agent (Beginner)**.

## What it does

Given a student's profile (background, known skills, and goal) and a
course catalogue (with prerequisites), the agent uses an LLM to reason
about the student's skill gaps and produce an ordered, justified
learning path — one course at a time, each with a specific reason tied
to that student.

## Why this is agentic, not hardcoded

There is no if/else logic mapping goals to courses. The full catalogue
and the student profile are given to the LLM in one prompt
(`build_prompt`), and the model itself reasons about:
- which skills the student is missing for their goal,
- which courses teach those skills,
- what order respects prerequisites,
- and why each course specifically helps this student.

The only fixed code is I/O (reading JSON, writing output files) and the
instructions given to the model — the recommendation itself is entirely
the LLM's reasoning.

## Project structure

```
course_agent/
├── course_recommendation_agent.py   # the agent
├── catalogue.json                    # sample course catalogue (13 courses)
├── student_profiles.json             # 4 sample student profiles
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Works with **any** of Groq, OpenAI, Anthropic, or Gemini — pick one:

```bash
pip install -r requirements.txt

# Default: Groq (free, no credit card needed)
export LLM_PROVIDER=groq
export GROQ_API_KEY="your-key"        # console.groq.com/keys

# OR OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# OR Anthropic
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OR Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="..."
```

No provider SDKs required — the agent talks to each API directly over
plain REST calls, so switching providers is just an environment
variable change.

## Usage

```bash
# Run all 4 sample profiles
python course_recommendation_agent.py

# Run just one profile
python course_recommendation_agent.py --profile "Aisha"

# Use your own catalogue / profiles file
python course_recommendation_agent.py --catalogue my_catalogue.json --profiles my_profiles.json
```

Output is written to `outputs/`:
- `<name>_path.md` — a human-readable learning path per student
- `all_recommendations.json` — combined structured output for all profiles

## Sample data included

- `catalogue.json` — 13 courses across programming, data science, and
  web development, each with prerequisites, level, and duration.
- `student_profiles.json` — 4 varied students (a complete beginner, a
  Python-literate aspiring ML engineer, a self-taught front-end learner,
  and an ML-experienced student wanting to specialize in NLP), so the
  reviewer can see the agent adapt its reasoning to different starting
  points and goals.

## Design tradeoffs & limitations

- **Single LLM call per student** rather than a multi-agent pipeline —
  kept intentionally simple for a Beginner-tier agent; the catalogue is
  small enough to fit entirely in one prompt without retrieval.
- **No persistent database** — catalogue and profiles are flat JSON
  files, which is sufficient at this scale and keeps setup to a single
  `pip install`.
- **Provider-agnostic LLM layer** — `call_llm()` supports Groq, OpenAI,
  Anthropic, and Gemini via plain REST calls (no extra SDKs needed),
  selected with `LLM_PROVIDER`. Groq is the default since it's free
  with no billing-gated quota, but any of the four work identically.
- **Known limitation**: for a much larger catalogue (hundreds of
  courses), the whole-catalogue-in-one-prompt approach would need to
  become retrieval-based (e.g. only surfacing courses relevant to the
  student's goal) to stay within context limits. With more time, I'd
  add a first LLM pass that filters the catalogue down before the
  reasoning pass.
