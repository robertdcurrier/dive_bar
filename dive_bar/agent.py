#!/usr/bin/env python3
"""Agent class for Dive Bar."""

from dive_bar.models import AgentConfig, Message

SYSTEM_TEMPLATE = """You are {name}. You are at a dive bar \
called {bar_name}. You are not an AI. You are a person.

About you: {backstory}

Your personality: {traits}.
How you talk: {speaking_style}.
Your drink: {drink}.

Always speak in FIRST PERSON as {name}. Say "I"
not "{name}". You are talking out loud, not
narrating. One or two short sentences max. Talk
casually -- use contractions, fragments, slang.
Talk about normal bar stuff: your day, complaints,
gossip, stories, sports, the weather, life.
Each reply must use completely fresh words and
phrasing. Vary your sentence openers every time.
Disagree, challenge, or change the subject.
Share a specific personal story or opinion that
only YOU would have based on your backstory.
Use vivid, original language. Surprise people.
Say your line out loud in first person and stop.
Keep it to spoken dialogue only."""

TOPIC_PROMPT = (
    "Name a random dive bar conversation topic "
    "in 2-5 words. Just the topic, nothing else."
    " Be specific and gritty. Examples: 'worst "
    "landlord stories', 'dumbest bar fights', "
    "'jobs that broke you', 'creepy regulars'."
)

TOPIC_CHANGE_TEMPLATE = (
    "{script}\n\n"
    "{name} completely drops the old subject and "
    "brings up {topic}. Do NOT reference anything "
    "from the previous conversation. Reply as "
    "{name} starting fresh on this topic. 1-2 "
    "sentences, first person, no name prefix."
)

CHARS_PER_TOKEN = 4


class Agent:
    """An AI agent with a personality at the bar."""

    def __init__(
        self,
        config: AgentConfig,
        bar_name: str,
        max_context: int = 4096,
        max_tokens: int = 200,
    ):
        self.config = config
        self.bar_name = bar_name
        self.max_context = max_context
        self.max_tokens = max_tokens
        self.system_prompt = self._build_system()

    def _build_system(self) -> str:
        """Construct the system prompt."""
        traits = ", ".join(
            self.config.personality_traits
        )
        return SYSTEM_TEMPLATE.format(
            name=self.config.name,
            bar_name=self.bar_name,
            backstory=self.config.backstory,
            traits=traits,
            speaking_style=(
                self.config.speaking_style
            ),
            drink=self.config.drink,
        )

    def build_topic_prompt(self) -> list[dict]:
        """Build messages to generate a random topic."""
        return [
            {
                "role": "system",
                "content": TOPIC_PROMPT,
            },
            {
                "role": "user",
                "content": "Give me a topic.",
            },
        ]

    def build_messages(
        self,
        history: list[Message],
        new_topic: str | None = None,
    ) -> list[dict]:
        """Build chat messages for the LLM.

        Packs conversation as a script in one user
        message so the model continues naturally.
        When new_topic is set, instructs the agent
        to change the subject.
        """
        system_msg = {
            "role": "system",
            "content": self.system_prompt,
        }
        sys_tokens = self._estimate_tokens(
            self.system_prompt
        )
        budget = (
            self.max_context
            - sys_tokens
            - self.max_tokens
            - 100  # safety margin
        )
        script = self._build_script(
            history, budget
        )
        name = self.config.name
        if new_topic:
            content = TOPIC_CHANGE_TEMPLATE.format(
                script=script,
                topic=new_topic,
                name=name,
            )
        else:
            content = (
                f"{script}\n\n"
                f"Now reply as {name}, in first "
                f"person. Stay on the current topic "
                f"or riff on what was just said. "
                f"1-2 sentences. No name prefix, "
                f"no narration."
            )
        user_msg = {
            "role": "user",
            "content": content,
        }
        return [system_msg, user_msg]

    def _build_script(
        self,
        history: list[Message],
        budget: int,
    ) -> str:
        """Build a script of recent conversation."""
        lines = []
        used = 0
        for msg in reversed(history):
            line = f"{msg.agent_name}: {msg.content}"
            tokens = self._estimate_tokens(line)
            if used + tokens > budget:
                break
            lines.append(line)
            used += tokens
        lines.reverse()
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text) // CHARS_PER_TOKEN + 1
