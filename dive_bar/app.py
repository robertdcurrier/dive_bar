#!/usr/bin/env python3
"""Main Textual application for Dive Bar."""

import random
import re
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header
from textual.worker import Worker, WorkerState

from dive_bar.agent import Agent
from dive_bar.bartender import Bartender
from dive_bar.db import Database
from dive_bar.diversity import (
    DiversityResult,
    compute_diversity_score,
)
from dive_bar.inference import InferenceEngine
from dive_bar.models import AppConfig, Message
from dive_bar.widgets.agent_sidebar import (
    AgentSidebar,
)
from dive_bar.widgets.chat_panel import ChatPanel
from dive_bar.widgets.controls import (
    BarControls,
    PauseToggled,
    SpeedChanged,
    StrangerMessage,
)


class DiveBarApp(App):
    """The Dive Bar TUI application."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr auto;
        grid-rows: 1fr auto;
    }
    #main-area {
        width: 100%;
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("p", "toggle_pause", "Pause"),
        Binding("q", "quit", "Quit"),
        Binding(
            "plus", "speed_up", "Speed +",
            show=False,
        ),
        Binding(
            "minus", "speed_down", "Speed -",
            show=False,
        ),
    ]

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.title = (
            f"{config.bar.name} -- Dive Bar"
        )
        self.history: list[Message] = []
        self.speed_mult = 1.0
        self._session_id = ""
        self._subject_count = 0
        self._recent_topics: list[str] = []
        self._opener = ""
        self._setup_components()

    def _setup_components(self):
        """Initialize core components."""
        self.engine = self._create_engine()
        self.engines: dict = {}
        self.house_engine = None
        if self.config.llm.mode == "api":
            self._create_api_engines()
        root = Path(__file__).resolve().parent.parent
        db_path = root / self.config.database.path
        self.db = Database(str(db_path))
        self.agents = self._create_agents()
        self.bartender = Bartender(
            [a.config for a in self.agents.values()],
            self.config.bar.tick_interval,
        )

    def _create_engine(self):
        """Pick inference engine based on config mode."""
        if self.config.llm.mode == "api":
            from dive_bar.api_engine import APIEngine
            return APIEngine(self.config.llm)
        return InferenceEngine(self.config.llm)

    def _create_api_engines(self):
        """Create per-agent API engines + house engine."""
        from dive_bar.api_engine import APIEngine
        for ac in self.config.agents:
            self.engines[ac.name] = APIEngine(
                self.config.llm
            )
        self.house_engine = APIEngine(
            self.config.llm
        )

    def _get_engine(self, name: str | None = None):
        """Route to the correct engine by agent name.

        Per-agent engine if available, else house
        engine, else single local engine.
        """
        if name and name in self.engines:
            return self.engines[name]
        if self.house_engine:
            return self.house_engine
        return self.engine

    def _create_agents(self) -> dict[str, Agent]:
        """Create Agent instances from config."""
        agents = {}
        limit = self.config.bar.max_agents
        for ac in self.config.agents[:limit]:
            agents[ac.name] = Agent(
                config=ac,
                bar_name=self.config.bar.name,
                max_context=self.config.llm.n_ctx,
                max_tokens=(
                    self.config.llm.generation.max_tokens
                ),
            )
        return agents

    def compose(self) -> ComposeResult:
        """Build the app layout."""
        yield Header()
        agent_info = [
            (a.config.name, a.config.drink)
            for a in self.agents.values()
        ]
        with Horizontal(id="main-area"):
            yield ChatPanel(id="chat")
            yield AgentSidebar(
                agent_info, id="sidebar"
            )
        yield BarControls(id="controls")
        yield Footer()

    def on_mount(self) -> None:
        """Start up: load model, begin conversation."""
        self._controls.set_status(
            "Loading model..."
        )
        self.run_worker(
            self._startup,
            thread=True,
            group="startup",
        )

    def _startup(self) -> str:
        """Load model in background thread."""
        if self.config.llm.mode == "local":
            root = (
                Path(__file__).resolve().parent.parent
            )
            model_path = (
                root / self.config.llm.model_path
            )
            self.config.llm.model_path = str(
                model_path
            )
            self.engine.config.model_path = str(
                model_path
            )
            self.engine.load_model()
        else:
            self._load_api_engines()
        self._opener = self._generate_opener()
        return "ready"

    def _load_api_engines(self):
        """Load all per-agent + house API engines."""
        for eng in self.engines.values():
            eng.load_model()
        if self.house_engine:
            self.house_engine.load_model()

    def on_worker_state_changed(
        self, event: Worker.StateChanged
    ) -> None:
        """Handle worker completion."""
        if (
            event.worker.group == "startup"
            and event.state == WorkerState.SUCCESS
        ):
            self._on_model_ready()

    def _on_model_ready(self):
        """Model loaded -- start the conversation."""
        self._session_id = self.db.start_session(
            bar_name=self.config.bar.name,
            agent_count=len(self.agents),
        )
        self._chat.add_system_message(
            "The bar is open. "
            "Patrons settle onto their stools..."
        )
        self._seed_conversation()
        self._controls.set_status("Bar is open.")
        interval = self.config.bar.tick_interval
        self.set_timer(interval, self._tick)

    OPENER_CATEGORIES = [
        "sex and hookups",
        "politics",
        "marriage",
        "kids and parenting",
        "pets",
        "girlfriends and boyfriends",
        "work complaints",
        "crazy news stories",
        "neighborhood gossip",
        "money problems",
        "bad dates",
        "family drama",
        "landlord horror stories",
        "worst coworkers",
        "celebrity gossip",
        "gas prices and inflation",
    ]

    def _generate_opener(self) -> str:
        """Ask the LLM for a bartender opener."""
        topic = random.choice(self.OPENER_CATEGORIES)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a bartender at a dive "
                    "bar. Write one casual sentence "
                    "about {topic} to kick off "
                    "tonight's conversation. Sound "
                    "natural and gruff. No quotes, "
                    "no narration. Under 15 words."
                ).format(topic=topic),
            },
            {
                "role": "user",
                "content": (
                    "Say something about {topic} "
                    "to get the regulars talking."
                ).format(topic=topic),
            },
        ]
        result = self._get_engine().generate(
            prompt, stop=["\n"], max_tokens=30,
        )
        text = result.content.strip().strip("'\"")
        return text or "Slow night. Somebody " \
            "say something interesting."

    def _seed_conversation(self):
        """Add the generated opener to history."""
        seed = Message(
            agent_name="Bartender",
            content=self._opener,
            turn_number=0,
            timestamp=time.time(),
        )
        self.history.append(seed)
        self._chat.add_message(
            "Bartender", seed.content,
            time.strftime("%H:%M"),
        )

    def _tick(self):
        """Orchestrator tick: pick next speaker."""
        if self.bartender.paused:
            self._schedule_next_tick()
            return
        last = (
            self.history[-1]
            if self.history
            else None
        )
        name = self.bartender.select_next(last)
        if name is None:
            self._schedule_next_tick()
            return
        self._sidebar.set_agent_status(
            name, "thinking"
        )
        self._controls.set_status(
            f"{name} is thinking..."
        )
        self.run_worker(
            lambda: self._generate_turn(name),
            thread=True,
            group="inference",
            exclusive=True,
        )

    def _generate_turn(self, name: str) -> dict:
        """Run inference for an agent (worker)."""
        agent = self.agents[name]
        max_subj = self.config.bar.max_subject_chat
        messages = self._build_turn_messages(
            agent, max_subj
        )
        stop = [
            f"{n}:" for n in self.agents
            if n != name
        ] + ["Bartender:", "A stranger:", "\n\n"]
        engine = self._get_engine(name)
        result = engine.generate(
            messages, stop=stop
        )
        content = self._clean_response(
            name, result.content
        )
        # Diversity check and regeneration loop
        div_cfg = self.config.diversity
        if div_cfg.enabled and content:
            content, regen_count = self._diversity_loop(
                name, agent, content, stop, div_cfg
            )
            if regen_count > 0:
                self._log_regeneration(
                    name, regen_count
                )
        score = self.bartender.get_score(
            name,
            self.history[-1]
            if self.history
            else None,
        )
        # Always record spoke to prevent loops
        last_speaker = (
            self.history[-1].agent_name
            if self.history else None
        )
        self.bartender.record_spoke(
            name, last_speaker
        )
        if not content:
            self.call_from_thread(
                self._on_empty_response, name
            )
            return {"name": name, "content": ""}
        turn = self.bartender.turn_number
        msg = Message(
            agent_name=name,
            content=content,
            turn_number=turn,
            timestamp=time.time(),
        )
        self.history.append(msg)
        self._log_to_db(
            name, content, result, score, turn
        )
        self.call_from_thread(
            self._display_message, name,
            content, turn,
        )
        return {"name": name, "content": content}

    def _build_turn_messages(
        self, agent, max_subj
    ) -> list[dict]:
        """Build messages, rotating topic if needed.

        When _subject_count hits max_subject_chat,
        generates a random topic via a short LLM call
        and tells the agent to change the subject.
        """
        if self._subject_count >= max_subj:
            topic = self._generate_topic(
                agent, agent.config.name
            )
            self._subject_count = 0
            return agent.build_messages(
                self.history, new_topic=topic
            )
        self._subject_count += 1
        return agent.build_messages(self.history)

    def _generate_topic(
        self, agent, name: str | None = None
    ) -> str:
        """Ask the LLM for a random bar topic."""
        prompt = agent.build_topic_prompt(
            self._recent_topics or None
        )
        result = self._get_engine(name).generate(
            prompt,
            stop=["\n"],
            max_tokens=20,
        )
        topic = result.content.strip().strip("'\"")
        if topic:
            self._recent_topics.append(topic)
            if len(self._recent_topics) > 5:
                self._recent_topics.pop(0)
        return topic

    def _diversity_loop(
        self,
        name: str,
        agent: Agent,
        content: str,
        stop: list[str],
        div_cfg,
    ) -> tuple[str, int]:
        """Check diversity and regenerate if needed.

        Returns (final_content, regen_count).
        """
        attempt = 0
        while attempt < div_cfg.max_retries:
            diversity = compute_diversity_score(
                content,
                self.history,
                name,
                window=div_cfg.window_size,
                threshold=div_cfg.threshold,
                ngram_min=div_cfg.ngram_min,
                ngram_max=div_cfg.ngram_max,
            )
            if diversity.passed:
                break
            feedback = self._build_rephrase_prompt(
                agent, content, diversity
            )
            result = self._get_engine(name).generate(
                feedback, stop=stop
            )
            content = self._clean_response(
                name, result.content
            )
            attempt += 1
            if not content:
                break
        return content, attempt

    def _build_rephrase_prompt(
        self,
        agent: Agent,
        original: str,
        diversity: DiversityResult,
    ) -> list[dict]:
        """Build prompt asking LLM to rephrase."""
        problems = "\n".join(
            f"- {p}" for p in diversity.problems[:3]
        )
        return [
            {
                "role": "system",
                "content": agent.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"You just said: \"{original}\"\n\n"
                    f"Problems:\n{problems}\n\n"
                    f"Say the same thing completely "
                    f"differently. Fresh phrasing. "
                    f"1-2 sentences."
                ),
            },
        ]

    def _log_regeneration(
        self, name: str, attempts: int
    ) -> None:
        """Log regeneration event to database."""
        self.db.log_regeneration(
            session_id=self._session_id,
            turn_number=self.bartender.turn_number,
            agent_name=name,
            attempt_count=attempts,
        )

    def _on_empty_response(self, name: str):
        """Handle empty generation -- just move on."""
        self._sidebar.set_agent_status(
            name, "idle"
        )
        self._schedule_next_tick()

    def _clean_response(
        self, name: str, text: str
    ) -> str:
        """Strip name prefixes and truncate at
        any dialogue pattern."""
        cleaned = text.strip()
        # Strip leading "Name:" / "Name N:" prefix
        cleaned = re.sub(
            r'^[A-Z][\w]*(\s+\d+)?:\s*',
            '', cleaned, count=1
        ).strip()
        # Truncate at any "Name:" mid-response
        match = re.search(
            r'\n\s*[A-Z][\w\s]*:', cleaned
        )
        if match:
            cleaned = cleaned[
                :match.start()
            ].strip()
        return cleaned

    def _log_to_db(
        self, name, content, result, score, turn
    ):
        """Log message to DuckDB."""
        gen = self.config.llm.generation
        agent_cfg = self.agents[name].config
        if self.config.llm.mode == "api":
            model_name = self.config.llm.api.model
        else:
            model_name = Path(
                self.config.llm.model_path
            ).stem
        self.db.log_message(
            session_id=self._session_id,
            turn_number=turn,
            agent_name=name,
            content=content,
            model_name=model_name,
            tokens_prompt=result.tokens_prompt,
            tokens_completion=(
                result.tokens_completion
            ),
            generation_time_ms=(
                result.generation_time_ms
            ),
            temperature=gen.temperature,
            top_p=gen.top_p,
            chattiness=agent_cfg.chattiness,
            score=score,
        )

    def _display_message(
        self, name: str, content: str, turn: int
    ):
        """Display message on the main thread."""
        ts = time.strftime("%H:%M")
        self._chat.add_message(name, content, ts)
        self._sidebar.set_agent_status(name, "idle")
        self._sidebar.update_stats(
            turn, self.speed_mult
        )
        self._controls.set_status("Bar is open.")
        self._schedule_next_tick()

    def _schedule_next_tick(self):
        """Schedule the next bartender tick."""
        interval = (
            self.config.bar.tick_interval
            / self.speed_mult
        )
        self.set_timer(interval, self._tick)

    @property
    def _chat(self) -> ChatPanel:
        return self.query_one("#chat", ChatPanel)

    @property
    def _sidebar(self) -> AgentSidebar:
        return self.query_one(
            "#sidebar", AgentSidebar
        )

    @property
    def _controls(self) -> BarControls:
        return self.query_one(
            "#controls", BarControls
        )

    def on_pause_toggled(
        self, event: PauseToggled
    ) -> None:
        """Handle pause toggle from controls."""
        self.bartender.paused = event.paused
        status = (
            "Paused." if event.paused
            else "Bar is open."
        )
        self._controls.set_status(status)

    def on_speed_changed(
        self, event: SpeedChanged
    ) -> None:
        """Handle speed change from controls."""
        self.speed_mult = max(
            0.25,
            min(4.0, self.speed_mult + event.delta),
        )
        self._sidebar.update_stats(
            self.bartender.turn_number,
            self.speed_mult,
        )

    def on_stranger_message(
        self, event: StrangerMessage
    ) -> None:
        """Handle injected stranger message."""
        turn = self.bartender.turn_number + 1
        msg = Message(
            agent_name="A stranger",
            content=event.text,
            turn_number=turn,
            timestamp=time.time(),
        )
        self.history.append(msg)
        ts = time.strftime("%H:%M")
        self._chat.add_message(
            "A stranger", event.text, ts
        )
        self.db.log_message(
            session_id=self._session_id,
            turn_number=turn,
            agent_name="A stranger",
            content=event.text,
        )

    def action_toggle_pause(self) -> None:
        """Keybinding: toggle pause."""
        self.bartender.paused = (
            not self.bartender.paused
        )
        status = (
            "Paused." if self.bartender.paused
            else "Bar is open."
        )
        self._controls.set_status(status)

    def action_speed_up(self) -> None:
        """Keybinding: increase speed."""
        self.speed_mult = min(
            4.0, self.speed_mult + 0.25
        )

    def action_speed_down(self) -> None:
        """Keybinding: decrease speed."""
        self.speed_mult = max(
            0.25, self.speed_mult - 0.25
        )

    def action_quit(self) -> None:
        """Clean shutdown."""
        if self._session_id:
            self.db.end_session(self._session_id)
        self.db.close()
        if self.engines:
            for eng in self.engines.values():
                eng.unload()
            if self.house_engine:
                self.house_engine.unload()
        else:
            self.engine.unload()
        self.exit()
