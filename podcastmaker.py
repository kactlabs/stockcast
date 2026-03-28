

"""
Stock Analysis Discussion System using AgentScope
Two agents discuss low-key US stocks with potential for 6-month growth

https://doc.agentscope.io/tutorial/workflow_multiagent_debate.html
March 25, 2026
"""

import asyncio
import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from typing import List

from pydantic import Field, BaseModel

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub

from llm import (
    LLMFactory,
    get_llm,
    fetch_content_from_url,
    process_instructions_with_url,
)

load_dotenv()

# Load stock symbols from external file (one symbol per line)
def _load_stock_symbols(path: str = "stocks.txt") -> List[str]:
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Stock symbols file not found: {path}")

SAMPLE_STOCKS = _load_stock_symbols()


def fetch_stock_research(symbols: List[str]) -> str:
    """Fetch supplementary stock research using the LLM provider from llm.py.

    Uses get_llm() to query the configured provider (ollama, openai, llama.cpp,
    gemini) for a quick research summary on the given symbols.  Falls back
    gracefully if the provider is unavailable.
    """
    try:
        llm_client = get_llm()
        prompt = (
            f"Provide a brief research summary for these stock symbols: "
            f"{', '.join(symbols)}. Include recent news, key metrics, and "
            f"growth catalysts. Keep it concise — 2-3 sentences per stock."
        )
        response = llm_client.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        logger.info("Fetched supplementary research via llm.py for: %s", ", ".join(symbols))
        return content
    except Exception as exc:
        logger.warning("Could not fetch supplementary research via llm.py: %s", exc)
        return ""


def fetch_stock_news_from_url(url: str) -> str:
    """Fetch stock-related content from a URL using llm.py's URL fetcher.

    Returns the extracted text or an empty string on failure.
    """
    result = fetch_content_from_url(url)
    if result["success"]:
        # Trim to a reasonable size for agent context
        content = result["content"]
        if len(content) > 3000:
            content = content[:3000] + "\n\n[Content truncated for length...]"
        return content
    logger.warning("URL fetch failed for %s: %s", url, result["error"])
    return ""

def _resolve_model_name() -> str:
    """Return the model name based on the active LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    # llama.cpp or unknown — model name is ignored by the server
    return "gpt-3.5-turbo"


def create_stock_analyst_agent(name: str, role: str) -> ReActAgent:
    """Create a stock analyst agent with specific role."""
    logger.info("Creating stock analyst agent: %s (%s)", name, role)
    
    prompt_file = f"prompts/{role}.txt"
    with open(prompt_file, "r") as f:
        sys_prompt = f.read().strip().format(name=name)
    
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name=_resolve_model_name(),
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            client_kwargs={
                "base_url": os.getenv("BASE_URL", "http://127.0.0.1:8080/v1")
            },
            generate_kwargs={
                "temperature": 0.7,
                "max_tokens": 300,
            }
        ),
        formatter=DashScopeMultiAgentFormatter(),
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def auto_commit_markdown_file(filename: str) -> None:
    """Commit a generated markdown file with a fixed commit message."""
    if os.getenv("COMMIT_CHANGES", "false").lower() != "true":
        logger.info("Auto-commit disabled (COMMIT_CHANGES != true)")
        return

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(repo_dir, filename))
    rel_file_path = os.path.relpath(file_path, repo_dir)

    try:
        inside_repo = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if inside_repo.returncode != 0:
            logger.warning("Skipping auto-commit: not inside a git repository")
            return

        add_result = subprocess.run(
            ["git", "-C", repo_dir, "add", "--", rel_file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if add_result.returncode != 0:
            logger.warning("Skipping auto-commit: git add failed: %s", add_result.stderr.strip())
            return

        commit_result = subprocess.run(
            [
                "git",
                "-C",
                repo_dir,
                "commit",
                "--only",
                "-m",
                "markdown generated",
                "--",
                rel_file_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_result.returncode == 0:
            logger.info("Auto-committed report: %s", filename)
        else:
            logger.warning("Auto-commit skipped or failed: %s", commit_result.stderr.strip())
    except FileNotFoundError:
        logger.warning("Skipping auto-commit: git is not installed")

# Create the two analyst agents
alex = create_stock_analyst_agent(
    "Alex", 
    "proposer"
)

jordan = create_stock_analyst_agent(
    "Jordan", 
    "validator"
)

# Create a moderator to manage the discussion and final decision
moderator = ReActAgent(
    name="Moderator",
    sys_prompt=(
        "You are a senior investment advisor moderating a stock analysis discussion. "
        "Two analysts will discuss low-key US stocks with 6-month growth potential. "
        "Your job is to: 1) Keep the discussion focused and productive, "
        "2) Determine when enough analysis has been done, "
        "3) Make the final decision on the top 2 stocks to recommend based on "
        "the discussion. Summarize the key points and provide clear reasoning "
        "for the final recommendations."
    ),
    model=OpenAIChatModel(
        model_name=_resolve_model_name(),
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        client_kwargs={
            "base_url": os.getenv("BASE_URL", "http://127.0.0.1:8080/v1")
        },
        generate_kwargs={
            "temperature": 0.6,
            "max_tokens": 400,
        }
    ),
    formatter=DashScopeMultiAgentFormatter(),
)


class StockRecommendation(BaseModel):
    """Final stock recommendation structure."""
    
    discussion_complete: bool = Field(
        description="Whether the discussion has covered enough analysis"
    )
    recommended_stocks: List[str] = Field(
        description="List of 2 recommended stock symbols",
        default=[]
    )
    reasoning: str = Field(
        description="Summary of reasoning for the recommendations",
        default=""
    )


def extract_text_content(content: object) -> str:
    """Extract readable text from AgentScope message content payloads."""
    if content is None:
        return ""

    if isinstance(content, str):
        return normalize_text_for_report(content)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return normalize_text_for_report(content["text"])
        return str(content)

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            extracted = extract_text_content(item)
            if extracted:
                parts.append(extracted)
        return normalize_text_for_report("\n\n".join(parts))

    return normalize_text_for_report(str(content))


def normalize_text_for_report(text: str) -> str:
    """Reduce escaped markdown artifacts and add structure-friendly spacing."""
    cleaned = text.strip()

    # Unescape common markdown characters that appear in model payload text.
    cleaned = cleaned.replace("\\#", "#").replace("\\*", "*")

    # Ensure major headings start on clean lines.
    cleaned = re.sub(r"\s+(#{2,6}\s)", r"\n\n\1", cleaned)
    cleaned = re.sub(r"\s+(\d+\.\s+\*\*)", r"\n\n\1", cleaned)

    # Promote common inline section labels to sub-headings for readability.
    cleaned = re.sub(r"\*\*(Analysis|Key Catalyst|Fundamentals|Technical Analysis|Opportunities|Risks|Risk Profile|Expected Return|Growth Catalysts|Market Position|Recent Positive Developments|Reasoning):\*\*\s*", r"\n\n#### \1\n", cleaned, flags=re.IGNORECASE)

    # Ensure bold blocks (**...**) that start a line get a blank line before them.
    cleaned = re.sub(r"(?<!\n)\n(\*\*)", r"\n\n\1", cleaned)

    # Convert asterisk bullets (* item) to dash bullets (- item) with proper line breaks.
    cleaned = re.sub(r"(?:^|\n)\s*\*\s+(?!\*)", r"\n- ", cleaned)

    # Keep dash bullets separated and readable.
    cleaned = re.sub(r"\s-\s\*\*", r"\n- **", cleaned)
    cleaned = re.sub(r"(?<!\n)\s-\s(?!\*\*)", r"\n- ", cleaned)

    # Standardize stock heading style where models output numbered lists.
    cleaned = re.sub(r"^###\s*(\d+)\.\s*", r"#### Stock \1\n", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^(\d+)\.\s+\*\*", r"#### Stock \1\n**", cleaned, flags=re.MULTILINE)

    # Collapse excessive spacing while keeping paragraph breaks.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


class StockDiscussionTracker:
    """Tracks stock discussion content and exports markdown using MarkIt."""

    def __init__(self) -> None:
        self.rounds: List[dict] = []
        self.start_time = datetime.now()
        self.final_recommendations: List[str] = []
        self.final_reasoning: str = ""
        self.total_rounds: int = 0

    def add_round(
        self,
        round_num: int,
        alex_text: str,
        jordan_text: str,
        moderator_text: str,
    ) -> None:
        """Add one discussion round to the report."""
        self.rounds.append(
            {
                "round_num": round_num,
                "alex": alex_text,
                "jordan": jordan_text,
                "moderator": moderator_text,
            }
        )

    def set_final_decision(
        self,
        recommendations: List[str],
        reasoning: str,
        total_rounds: int,
    ) -> None:
        """Set final recommendation metadata for the report."""
        self.final_recommendations = recommendations
        self.final_reasoning = reasoning
        self.total_rounds = total_rounds

    def _build_html_report(self) -> str:
        """Build a structured HTML report for MarkIt conversion."""
        generated_at = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        recommendations_html = "".join(
            f"<li><strong>{symbol}</strong></li>" for symbol in self.final_recommendations
        )

        rounds_html = ""
        for round_data in self.rounds:
            alex_text = html.escape(round_data["alex"]).replace("\n", "<br/>\n")
            jordan_text = html.escape(round_data["jordan"]).replace("\n", "<br/>\n")
            moderator_text = html.escape(round_data["moderator"]).replace("\n", "<br/>\n")
            rounds_html += f"""
            <section>
              <h2>Round {round_data['round_num']}</h2>
              <h3>Alex (Proposer)</h3>
              <p>{alex_text}</p>
              <h3>Jordan (Validator)</h3>
              <p>{jordan_text}</p>
              <h3>Moderator Assessment</h3>
              <p>{moderator_text}</p>
            </section>
            """

        final_reasoning = html.escape(self.final_reasoning).replace("\n", "<br/>\n")

        return f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Stock Analysis Discussion Report</title>
  </head>
  <body>
    <h1>Stock Analysis Discussion Report</h1>
    <p><strong>Generated on:</strong> {generated_at}</p>
    <p><strong>Discussion rounds:</strong> {self.total_rounds}</p>

    <h2>Executive Summary</h2>
    <p>
      This report summarizes a multi-agent discussion focused on low-key US stocks
      with potential 6-month growth. It captures proposals, risk validation, and
      final investment recommendations.
    </p>

    {rounds_html}

    <h2>Final Recommendations</h2>
    <ul>
      {recommendations_html}
    </ul>
        <p>{final_reasoning}</p>
    <hr/>
    <p><strong>Provider:</strong> {os.getenv('LLM_PROVIDER', 'llama.cpp')}<br/>
    <strong>Model:</strong> {_resolve_model_name()}</p>
  </body>
</html>
""".strip()

    def _build_markdown_fallback(self) -> str:
        """Fallback markdown when MarkIt is not available."""
        generated_at = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        lines: List[str] = [
            "[Home](index.md)",
            "",
            "# Stock Analysis Discussion Report",
            "",
            f"**Generated on:** {generated_at}",
            f"**Discussion rounds:** {self.total_rounds}",
            "",
            "## Executive Summary",
            "",
            "This report summarizes a multi-agent discussion focused on low-key US stocks with potential 6-month growth.",
            "",
        ]

        for round_data in self.rounds:
            lines.extend(
                [
                    f"## Round {round_data['round_num']}",
                    "",
                    "### Alex (Proposer)",
                    "",
                    round_data["alex"],
                    "",
                    "### Jordan (Validator)",
                    "",
                    round_data["jordan"],
                    "",
                    "### Moderator Assessment",
                    "",
                    round_data["moderator"],
                    "",
                ]
            )

        lines.extend(
            [
                "## Final Recommendations",
                "",
            ]
        )
        for symbol in self.final_recommendations:
            lines.append(f"- **{symbol}**")

        lines.extend(
            [
                "",
                self.final_reasoning,
                "",
                "---",
                "",
                f"**Provider:** {os.getenv('LLM_PROVIDER', 'llama.cpp')}  ",
                f"**Model:** {_resolve_model_name()}",
                "",
            ]
        )

        return "\n".join(lines)

    def _update_index(self, filename: str) -> None:
        """Append a link to the generated report in index.md."""
        index_path = "index.md"
        title = ", ".join(self.final_recommendations) if self.final_recommendations else "Stock Analysis"
        title = f"{title} — {self.start_time.strftime('%Y %b %d')}"
        entry = f"  * [{title}]({filename})\n"
        try:
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if filename not in content:
                    with open(index_path, "a", encoding="utf-8") as f:
                        f.write(entry)
            else:
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write(f"# StockCast\n\nStock Market Podcast by Agents\n\n## Category :\n\n{entry}")
            logger.info("Updated index.md with: %s", filename)
        except Exception as exc:
            logger.warning("Could not update index.md: %s", exc)

    def _prepend_home_link(self, filename: str) -> None:
        """Ensure [Home](index.md) is at the top of the file."""
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.startswith("[Home](index.md)"):
            with open(filename, "w", encoding="utf-8") as f:
                f.write("[Home](index.md)\n\n" + content)

    def export_report(self, output_filename: str) -> str:
        """Export report as markdown using MarkIt; fallback to native markdown."""
        html_report = self._build_html_report()
        markit_bin = shutil.which("markit")
        npx_bin = shutil.which("npx")

        command: List[str] | None = None
        if markit_bin:
            command = [markit_bin]
        elif npx_bin:
            command = [npx_bin, "-y", "markit-ai"]

        if command:
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", encoding="utf-8", delete=False
                ) as temp_file:
                    temp_file.write(html_report)
                    temp_path = temp_file.name

                completed = subprocess.run(
                    [*command, temp_path, "-q", "-o", output_filename],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.returncode == 0:
                    self._prepend_home_link(output_filename)
                    self._update_index(output_filename)
                    return "markit"

                logger.warning(
                    "MarkIt failed with exit code %s: %s",
                    completed.returncode,
                    completed.stderr.strip(),
                )
            except Exception as exc:
                logger.warning("MarkIt conversion failed: %s", exc)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        markdown_content = self._build_markdown_fallback()
        with open(output_filename, "w", encoding="utf-8") as file_handle:
            file_handle.write(markdown_content)
        self._update_index(output_filename)
        return "fallback"


def _parse_moderator_response(response: Msg, round_num: int = 0) -> dict:
    """Extract structured data from moderator response, with text fallback.

    Returns dict with keys: discussion_complete, recommended_stocks, reasoning.
    """
    metadata = getattr(response, "metadata", None) or {}

    # If structured output worked, use it directly
    if metadata.get("recommended_stocks"):
        return {
            "discussion_complete": metadata.get("discussion_complete", True),
            "recommended_stocks": metadata["recommended_stocks"],
            "reasoning": metadata.get("reasoning", ""),
        }

    # Fallback: parse stock symbols from the text content
    text = extract_text_content(response.content)
    # Look for uppercase 1-5 letter symbols that look like tickers
    symbols = re.findall(r"\b([A-Z]{1,5})\b", text)
    # Filter out common English words that look like tickers
    noise = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
        "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS",
        "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET",
        "LET", "SAY", "SHE", "TOO", "USE", "YES", "YET", "TOP", "TWO",
        "KEY", "LOW", "HIGH", "STEP", "WITH", "THIS", "THAT", "FROM",
        "HAVE", "BEEN", "WILL", "EACH", "MAKE", "LIKE", "LONG", "LOOK",
        "MANY", "SOME", "THEM", "THAN", "OVER", "SUCH", "TAKE", "INTO",
        "JUST", "ALSO", "VERY", "WHEN", "WHAT", "YOUR", "BOTH", "AFTER",
        "BASED", "FINAL", "STOCK", "ROUND", "COULD", "WOULD", "THEIR",
        "ABOUT", "WHICH", "THESE", "OTHER", "BEING", "FIRST", "GIVEN",
    }
    tickers = list(dict.fromkeys(s for s in symbols if s not in noise))

    is_complete = bool(tickers) or "final" in text.lower() or round_num >= 3
    recommendations = tickers[:2]  # Take top 2

    logger.info(
        "Structured output unavailable; parsed tickers from text: %s",
        recommendations,
    )
    return {
        "discussion_complete": is_complete and len(recommendations) >= 2,
        "recommended_stocks": recommendations,
        "reasoning": text,
    }


async def run_stock_analysis_discussion() -> None:
    """Run the stock analysis discussion between two agents."""
    round_num = 0
    max_rounds = 4  # Limit discussion to prevent infinite loops
    
    logger.info("Starting stock analysis discussion...")
    logger.info("Alex (Proposer) and Jordan (Validator) are analyzing low-key US stocks")
    tracker = StockDiscussionTracker()
    
    # Initial context message
    initial_context = (
        "Let's discuss low-key US stocks that could provide significant returns "
        "within the next 6 months. Focus on undervalued companies with upcoming "
        "catalysts, strong fundamentals, or emerging market opportunities. "
        f"Consider these sectors and examples: {', '.join(SAMPLE_STOCKS[:8])}"
    )
    
    # Fetch supplementary research via llm.py's multi-provider LLM support
    logger.info("Fetching supplementary stock research via llm.py...")
    supplementary_research = fetch_stock_research(SAMPLE_STOCKS[:4])
    if supplementary_research:
        initial_context += (
            f"\n\nSupplementary research data:\n{supplementary_research}"
        )
    
    while round_num < max_rounds:
        round_num += 1
        logger.info("=== Discussion Round %d ===", round_num)
        
        # Call each agent independently to maintain user/assistant alternation.
        # Pass the previous agent's output as context in a user-role message.
        if round_num == 1:
            logger.info("Alex is proposing initial stock ideas...")
            alex_prompt = initial_context + " Please propose 1-2 promising stocks with detailed analysis."
        else:
            alex_prompt = (
                "Continue the discussion. Propose additional stocks or defend previous suggestions with more analysis.\n\n"
                f"Previous round moderator feedback: {extract_text_content(moderator_response.content)}"
            )
        
        alex_response = await alex(Msg("user", alex_prompt, "user"))
        alex_text = extract_text_content(alex_response.content)
        
        logger.info("Jordan is validating and responding...")
        jordan_prompt = (
            f"Alex proposed the following:\n\n{alex_text}\n\n"
            "Analyze the proposed stocks. Provide validation, additional facts, or counter-arguments. "
            "What are the risks and opportunities?"
        )
        jordan_response = await jordan(Msg("user", jordan_prompt, "user"))
        jordan_text = extract_text_content(jordan_response.content)
        
        # Moderator evaluates
        logger.info("Moderator is evaluating the discussion...")
        moderator_prompt = (
            f"Round {round_num} discussion summary:\n\n"
            f"Alex (Proposer):\n{alex_text}\n\n"
            f"Jordan (Validator):\n{jordan_text}\n\n"
            "Based on the discussion so far, do we have enough analysis to make final recommendations? "
            "If yes, provide your top 2 stock recommendations with reasoning. "
            "If no, indicate that more discussion is needed."
        )
        moderator_response = await moderator(
            Msg("user", moderator_prompt, "user"),
            structured_model=StockRecommendation
        )
        
        parsed = _parse_moderator_response(moderator_response, round_num)
        
        if parsed["discussion_complete"]:
            recommendations = parsed["recommended_stocks"]
            reasoning = parsed["reasoning"]

            tracker.add_round(
                round_num=round_num,
                alex_text=extract_text_content(alex_response.content),
                jordan_text=extract_text_content(jordan_response.content),
                moderator_text=extract_text_content(moderator_response.content),
            )
            tracker.set_final_decision(recommendations, reasoning, round_num)
            
            logger.info("Discussion complete after %d rounds", round_num)
            print(f"\n🎯 FINAL STOCK RECOMMENDATIONS:")
            print(f"Recommended Stocks: {', '.join(recommendations)}")
            print(f"Reasoning: {reasoning}")
            break
        else:
            tracker.add_round(
                round_num=round_num,
                alex_text=extract_text_content(alex_response.content),
                jordan_text=extract_text_content(jordan_response.content),
                moderator_text=extract_text_content(moderator_response.content),
            )
            logger.info("Moderator says more discussion needed. Continuing...")
    
    if round_num >= max_rounds:
        logger.info("Maximum rounds reached. Forcing final recommendation...")
        final_response = await moderator(
            Msg(
                "user",
                "Time is up. Based on all the discussion, provide your final 2 stock recommendations now.",
                "user"
            ),
            structured_model=StockRecommendation
        )
        parsed = _parse_moderator_response(final_response, round_num)
        recommendations = parsed["recommended_stocks"]
        reasoning = parsed["reasoning"]
        tracker.set_final_decision(recommendations, reasoning, round_num)
        print(f"\n🎯 FINAL STOCK RECOMMENDATIONS (Time Limit):")
        print(f"Recommended Stocks: {', '.join(recommendations)}")
        print(f"Reasoning: {reasoning}")

    date_str = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        filename = f"stock_analysis_report_{date_str}_{index:03d}.md"
        if not os.path.exists(filename):
            break
        index += 1
    export_mode = tracker.export_report(filename)

    if export_mode == "markit":
        logger.info("Report generated with MarkIt and saved to: %s", filename)
        print(f"\n📄 Human-readable report saved via MarkIt: {filename}")
    else:
        logger.info("Report generated with fallback markdown and saved to: %s", filename)
        print(f"\n📄 Report saved with fallback markdown (MarkIt unavailable): {filename}")

    auto_commit_markdown_file(filename)


if __name__ == "__main__":
    asyncio.run(run_stock_analysis_discussion())