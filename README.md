# AgentScope Stock Analysis

A multi-agent system built with [AgentScope](https://doc.agentscope.io/) where AI analysts debate and recommend undervalued US stocks with 6-month growth potential. Reports are generated automatically as markdown.

## How It Works

Three agents collaborate in a structured, multi-round discussion (up to 4 rounds):

1. **Alex (Proposer)** — Identifies and pitches promising low-key stocks with detailed analysis
2. **Jordan (Validator)** — Fact-checks proposals, surfaces risks, and provides counter-arguments
3. **Moderator** — Steers the discussion, decides when enough analysis is done, and delivers the final top-2 recommendations

The system optionally fetches supplementary research via a configurable LLM provider (`llm.py`) before the discussion begins.

## Features

- Multi-round debate with automatic convergence or a hard cap at 4 rounds
- Supplementary stock research fetched through a pluggable LLM backend
- URL content extraction (BeautifulSoup) for injecting external data
- Structured output via Pydantic with a text-parsing fallback
- Report export through MarkIt (HTML → Markdown) with a native markdown fallback
- Optional auto-commit of generated reports (`COMMIT_CHANGES=true`)

## Supported LLM Providers

Configured via `LLM_PROVIDER` in `.env`. The stock discussion agents use AgentScope's `OpenAIChatModel` pointed at `BASE_URL`, while `llm.py` provides a separate adapter layer for supplementary research:

| Provider | Env Vars | Notes |
|---|---|---|
| llama.cpp | `BASE_URL` | Default. Server must be running locally |
| Ollama | `OLLAMA_MODEL` | Uses `langchain-ollama` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` | Uses `langchain-openai` |
| Gemini | `GOOGLE_API_KEY`, `GEMINI_MODEL` | Uses `langchain-google-genai` |

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy and configure your environment:
```bash
cp .env.example .env
```

Key variables in `.env`:
```bash
BASE_URL=http://127.0.0.1:8080/v1   # AgentScope model endpoint
LLM_PROVIDER=llama.cpp              # ollama | openai | llama.cpp | gemini
COMMIT_CHANGES=false                 # auto-commit reports to git
```

3. (Optional) Install MarkIt for cleaner report formatting:
```bash
npm install -g markit-ai
# or use npx — the script detects it automatically
```

4. Run:
```bash
python stock_symbol_analysis.py
```

## Generated Reports

Reports are saved as `stock_analysis_report_YYYYMMDD_NNN.md` with auto-incrementing sequence numbers. Each report contains:

- Executive summary
- Full round-by-round discussion transcript (Alex → Jordan → Moderator)
- Final top-2 stock recommendations with reasoning

If MarkIt is available the HTML discussion is converted to clean markdown; otherwise a native markdown fallback is written directly.

## Project Structure

```
├── stock_symbol_analysis.py        # Main multi-agent discussion script
├── stock_symbol_analysis_vscode.py # Alternate VS Code variant (cricket commentary demo)
├── llm.py                          # LLM adapter layer (Ollama, OpenAI, llama.cpp, Gemini)
├── requirements.txt
├── .env.example
└── stock_analysis_report_*.md      # Generated reports
```

## Customization

- Agent personalities: edit `sys_prompt` strings in `create_stock_analyst_agent()` and the moderator block
- Discussion depth: change `max_rounds` in `run_stock_analysis_discussion()`
- Model parameters: adjust `temperature` / `max_tokens` per agent
- Sample stock universe: update the `SAMPLE_STOCKS` list
- Report template: modify `StockDiscussionTracker._build_html_report()` or `_build_markdown_fallback()`
