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
Each reply must use completely fresh phrasing.
Vary your sentence openers every time.
Talk about YOURSELF -- your own experiences, your
own day, your own opinions. React directly to the
last speaker: argue, ask a question, crack a joke
at their expense, or call them out.
Use vivid, original language.
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
MAX_SCRIPT_LINES = 10


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
        last = (
            history[-1].agent_name
            if history else "them"
        )
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
                f"person. React directly to {last}"
                f" -- agree, disagree, ask them "
                f"something, or roast them. 1-2 "
                f"sentences. No name prefix, "
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
        """Build a script of recent conversation.

        Caps at MAX_SCRIPT_LINES to prevent echo
        templates from accumulating in the context.
        """
        recent = history[-MAX_SCRIPT_LINES:]
        lines = []
        used = 0
        for msg in reversed(recent):
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
