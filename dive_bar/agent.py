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
NEVER repeat or rephrase what someone just said.
NEVER say "I agree" or "I completely agree".
Have your OWN take. Push back, joke around, tell
a story, or go on a tangent. Be a real person
with opinions, not a yes-man.
Do NOT narrate or describe actions. Do NOT write
in third person. Do NOT be an assistant. Do NOT
use emojis. Do NOT invent new characters.
Just say your line out loud and stop."""

TOPIC_PROMPT = (
    "Name a random dive bar conversation topic "
    "in 2-5 words. Just the topic, nothing else."
    " Be specific and gritty. Examples: 'worst "
    "landlord stories', 'dumbest bar fights', "
    "'jobs that broke you', 'creepy regulars'."
)

TOPIC_CHANGE_TEMPLATE = (
    "{script}\n\n"
    "{name} gets bored and brings up {topic} "
    "like they just thought of it. Reply as "
    "{name} casually pivoting to this. 1-2 "
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
                f"person. Just the dialogue, 1-2 "
                f"sentences. No name prefix, no "
                f"narration."
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
