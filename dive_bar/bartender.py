#!/usr/bin/env python3
"""Bartender orchestrator for Dive Bar.

Decides which agent speaks next using a weighted
scoring algorithm. No LLM calls -- pure Python.
"""

import random
import time

from dive_bar.models import AgentConfig, Message

# Default scoring weights
W_TIME = 0.35
W_CHAT = 0.25
W_ADDR = 0.30
W_RAND = 0.10

# Max seconds before time_factor saturates at 1.0
TIME_CAP = 60.0

# Turns of silence before an agent gets a boost
SILENCE_THRESHOLD = 8
SILENCE_BOOST = 0.25

# Cooldown multiplier (tick_interval * this)
COOLDOWN_MULT = 1.5

# Max consecutive turns for the same A->B pair
# before deterministic addressing is suppressed
MAX_PAIR_STREAK = 2


class Bartender:
    """Orchestrator that selects the next speaker."""

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        tick_interval: float = 2.0,
    ):
        self.agents = {
            a.name: a for a in agent_configs
        }
        self.tick_interval = tick_interval
        self.cooldown = tick_interval * COOLDOWN_MULT
        self.last_spoke: dict[str, float] = {}
        self.last_spoke_turn: dict[str, int] = {}
        self.turn_number = 0
        self.paused = False
        self._pair_history: list[tuple[str, str]] = []

    def select_next(
        self,
        last_message: Message | None,
    ) -> str | None:
        """Pick the next agent to speak.

        If the last message addresses an agent by name,
        that agent speaks next (deterministic). Otherwise
        falls back to weighted scoring.
        """
        if self.paused:
            return None
        addressed = self._find_addressed(
            last_message
        )
        if addressed and not self._pair_locked(
            last_message.agent_name, addressed
        ):
            return addressed
        eligible = self._get_eligible()
        if not eligible:
            return None
        scores = {
            name: self._score_agent(
                name, last_message
            )
            for name in eligible
        }
        winner = max(scores, key=scores.get)
        return winner

    def _find_addressed(
        self,
        last_message: Message | None,
    ) -> str | None:
        """Check if last message names an agent.

        Returns the addressed agent's name if found
        and eligible, else None. Skips the speaker.
        """
        if last_message is None:
            return None
        content = last_message.content.lower()
        speaker = last_message.agent_name
        now = time.time()
        for name in self.agents:
            if name == speaker:
                continue
            last = self.last_spoke.get(name, 0.0)
            if now - last < self.cooldown:
                continue
            if name.lower() in content:
                return name
        return None

    def _pair_locked(
        self, speaker: str, responder: str
    ) -> bool:
        """True if this pair has been ping-ponging.

        Checks if the last MAX_PAIR_STREAK pairs all
        involve the same two agents in either direction
        (A->B or B->A both count).
        """
        dyad = frozenset((speaker, responder))
        tail = self._pair_history[-MAX_PAIR_STREAK:]
        if len(tail) < MAX_PAIR_STREAK:
            return False
        return all(
            frozenset(p) == dyad for p in tail
        )

    def record_spoke(
        self, agent_name: str,
        last_speaker: str | None = None,
    ):
        """Record that an agent just spoke."""
        self.last_spoke[agent_name] = time.time()
        self.last_spoke_turn[agent_name] = (
            self.turn_number
        )
        self.turn_number += 1
        if last_speaker:
            self._pair_history.append(
                (last_speaker, agent_name)
            )
            # Keep bounded
            if len(self._pair_history) > 20:
                self._pair_history = (
                    self._pair_history[-10:]
                )

    def get_score(
        self,
        agent_name: str,
        last_message: Message | None,
    ) -> float:
        """Public access to an agent's score."""
        return self._score_agent(
            agent_name, last_message
        )

    def _get_eligible(self) -> list[str]:
        """Return agents not in cooldown."""
        now = time.time()
        eligible = []
        for name in self.agents:
            last = self.last_spoke.get(name, 0.0)
            if now - last >= self.cooldown:
                eligible.append(name)
        return eligible

    def _score_agent(
        self,
        name: str,
        last_message: Message | None,
    ) -> float:
        """Compute selection score for one agent."""
        agent = self.agents[name]
        t_factor = self._time_factor(name)
        a_factor = self._addressed_factor(
            name, agent, last_message
        )
        s_boost = self._silence_boost(name)
        return (
            W_TIME * t_factor
            + W_CHAT * agent.chattiness
            + W_ADDR * a_factor
            + W_RAND * random.random()
            + s_boost
        )

    def _time_factor(self, name: str) -> float:
        """How long since this agent last spoke."""
        last = self.last_spoke.get(name, 0.0)
        if last == 0.0:
            return 1.0
        elapsed = time.time() - last
        return min(elapsed / TIME_CAP, 1.0)

    def _silence_boost(self, name: str) -> float:
        """Boost agents who haven't spoken in a while."""
        last_turn = self.last_spoke_turn.get(name, 0)
        gap = self.turn_number - last_turn
        if gap >= SILENCE_THRESHOLD:
            return SILENCE_BOOST
        return 0.0

    def _addressed_factor(
        self,
        name: str,
        agent: AgentConfig,
        last_message: Message | None,
    ) -> float:
        """Was this agent mentioned in the last msg?"""
        if last_message is None:
            return 0.0
        if last_message.agent_name == name:
            return 0.0
        if name.lower() in last_message.content.lower():
            return agent.responsiveness
        return 0.0
