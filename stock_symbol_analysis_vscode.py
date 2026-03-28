

"""
https://doc.agentscope.io/tutorial/workflow_multiagent_debate.html
March 25, 2026

https://doc.agentscope.io/tutorial/task_model.html#

https://deepwiki.com/modelscope/agentscope/3.5-ollama-and-local-models
"""


import asyncio
import logging
import random
import os
from dotenv import load_dotenv

from pydantic import Field, BaseModel

from agentscope.agent import ReActAgent
from agentscope.formatter import (
    DashScopeMultiAgentFormatter,
)
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub


load_dotenv()

# Possible ball outcomes with weights
OUTCOMES = [
    ("dot ball, 0 runs", 30),
    ("1 run, quick single", 25),
    ("2 runs, good running between the wickets", 15),
    ("3 runs, excellent placement", 5),
    ("FOUR! Boundary through the covers", 12),
    ("SIX! Smashed over the ropes", 5),
    ("WICKET! Batsman is OUT", 8),
]


def random_outcome() -> str:
    """Pick a weighted random outcome for a delivery."""
    labels, weights = zip(*OUTCOMES)
    return random.choices(labels, weights=weights, k=1)[0]


# Create two commentator agents, Harsha and Ravi, who will share live commentary.
def create_commentator_agent(name: str, style: str) -> ReActAgent:
    """Get a commentator agent."""
    logger.info("Creating commentator agent: %s", name)
    return ReActAgent(
        name=name,
        sys_prompt=(
            f"You are {name}, a live cricket commentator. {style} "
            "You are commentating on a single-over cricket match. "
            "Provide engaging, descriptive commentary (2-4 sentences). "
            "Add excitement, describe the shot/delivery, mention field placements when relevant. "
            "React to the other commentator's remarks naturally, like a real commentary pair. "
            "Build narrative tension and celebrate great moments with enthusiasm."
        ),
        model=OpenAIChatModel(
            model_name="gpt-3.5-turbo",  # dummy name (llama.cpp ignores it)
            client_kwargs={
                "base_url": os.getenv("BASE_URL", "http://127.0.0.1:8085/v1")
            },
            generate_kwargs={
                "temperature": 0.8,
                "max_tokens": 150,  # Increased for more verbose commentary
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

# Suppress HTTP request logs from httpx and other HTTP clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

harsha, ravi = [
    create_commentator_agent(
        "Harsha",
        "You are enthusiastic and love storytelling. You bring energy and excitement. "
        "Use vivid descriptions, mention the crowd's reaction, and paint the scene dramatically. "
        "You love to build up the moment and use colorful language.",
    ),
    create_commentator_agent(
        "Ravi",
        "You are analytical and technical. You focus on technique, field placements, and stats. "
        "Explain the bowler's strategy, batting technique, and tactical aspects. "
        "You provide expert insights and break down the technical elements of the game.",
    ),
]


# Scorecard tracking
class Scorecard(BaseModel):
    """Tracks the innings score."""

    runs: int = Field(default=0, description="Total runs scored")
    wickets: int = Field(default=0, description="Total wickets fallen")
    balls: int = Field(default=0, description="Balls bowled")


async def run_cricket_commentary() -> None:
    """Run a single-over cricket commentary with two commentators."""
    score = Scorecard()
    logger.info("Starting single-over cricket commentary...")
    logger.info("Harsha and Ravi are at the mic. Let's go!")

    for ball_num in range(1, 7):
        # Check if all out (10 wickets) — unlikely in 1 over but let's be correct
        if score.wickets >= 10:
            logger.info("All out! Innings over.")
            break

        outcome = random_outcome()
        score.balls = ball_num

        # Parse runs from outcome
        if "WICKET" in outcome:
            score.wickets += 1
        elif "SIX" in outcome:
            score.runs += 6
        elif "FOUR" in outcome:
            score.runs += 4
        elif "3 runs" in outcome:
            score.runs += 3
        elif "2 runs" in outcome:
            score.runs += 2
        elif "1 run" in outcome:
            score.runs += 1

        ball_info = (
            f"Ball {ball_num}/6 — {outcome}. "
            f"Score: {score.runs}/{score.wickets} after {score.balls} balls."
        )
        logger.info(ball_info)

        # Both commentators react to the ball inside MsgHub so they see each other's remarks
        async with MsgHub(participants=[harsha, ravi]):
            # logger.info("Harsha is commentating...")
            await harsha(
                Msg("user", ball_info, "user"),
            )
            # logger.info("Ravi is commentating...")
            await ravi(
                Msg(
                    "user",
                    f"Add your perspective on this delivery: {ball_info}",
                    "user",
                ),
            )

    logger.info(
        "Over complete! Final score: %d/%d in %d balls.",
        score.runs,
        score.wickets,
        score.balls,
    )
    print(
        f"\n🏏 Final Score: {score.runs}/{score.wickets} in {score.balls} balls."
    )


asyncio.run(run_cricket_commentary())