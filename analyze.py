#!/usr/bin/env python3
"""Conversation analysis CLI for Dive Bar.

Reads the DuckDB database and produces reports focused
on detecting echo/repetition problems across agents.
"""

import argparse
import re
import string
from collections import Counter, defaultdict

import duckdb
from rich.console import Console
from rich.table import Table

DEFAULT_DB = "data/dive_bar.duckdb"

STOP_WORDS = frozenset({
    "i", "me", "my", "we", "you", "your", "he", "she",
    "it", "they", "them", "the", "a", "an", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "am", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will",
    "just", "not", "no", "so", "if", "that", "this",
    "what", "when", "how", "all", "up", "out", "about",
    "like", "got", "get", "go", "can", "would", "could",
    "should", "there", "here", "from", "its", "than",
    "into", "over", "some", "then", "too", "very",
    "dont", "im", "ive", "thats", "yeah", "oh",
})


def open_db(path: str) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection."""
    return duckdb.connect(path, read_only=True)


def fetch_messages(
    con: duckdb.DuckDBPyConnection,
    session_id: str | None = None,
) -> list[dict]:
    """Query messages, optionally filtered by session."""
    query = (
        "SELECT message_id, session_id, "
        "turn_number, agent_name, content, "
        "created_at, tokens_prompt, "
        "tokens_completion, generation_time_ms "
        "FROM messages"
    )
    params = []
    if session_id:
        query += (
            " WHERE session_id LIKE ? || '%'"
        )
        params.append(session_id)
    query += " ORDER BY created_at, turn_number"
    rows = con.execute(query, params).fetchall()
    cols = [
        "message_id", "session_id", "turn_number",
        "agent_name", "content", "created_at",
        "tokens_prompt", "tokens_completion",
        "generation_time_ms",
    ]
    return [dict(zip(cols, r)) for r in rows]


def fetch_sessions(
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Query all sessions."""
    rows = con.execute(
        "SELECT session_id, started_at, ended_at, "
        "bar_name, agent_count, config_hash "
        "FROM sessions ORDER BY started_at"
    ).fetchall()
    cols = [
        "session_id", "started_at", "ended_at",
        "bar_name", "agent_count", "config_hash",
    ]
    return [dict(zip(cols, r)) for r in rows]


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stops."""
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    words = text.split()
    return [w for w in words if w not in STOP_WORDS]


def extract_ngrams(
    words: list[str], n: int
) -> list[tuple[str, ...]]:
    """Return contiguous n-grams from word list."""
    if len(words) < n:
        return []
    return [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]


def dedup_ngrams(
    phrases: dict[tuple[str, ...], set[str]],
) -> dict[tuple[str, ...], set[str]]:
    """Remove ngrams that are subsets of longer ones."""
    sorted_keys = sorted(
        phrases.keys(), key=len, reverse=True
    )
    kept: dict[tuple[str, ...], set[str]] = {}
    for gram in sorted_keys:
        gram_str = " ".join(gram)
        is_sub = any(
            gram_str in " ".join(k)
            for k in kept
            if len(k) > len(gram)
        )
        if not is_sub:
            kept[gram] = phrases[gram]
    return kept


def detect_echoes(
    messages: list[dict],
) -> dict[str, list]:
    """Cross-agent ngram analysis for echoes.

    Finds 3-6 word phrases used by 2+ different
    agents 3+ times total.
    """
    agent_ngrams: dict[
        tuple[str, ...], set[str]
    ] = defaultdict(set)
    ngram_counts: Counter = Counter()
    for msg in messages:
        words = tokenize(msg["content"])
        agent = msg["agent_name"]
        for n in range(3, 7):
            for gram in extract_ngrams(words, n):
                agent_ngrams[gram].add(agent)
                ngram_counts[gram] += 1
    shared = {
        gram: agents
        for gram, agents in agent_ngrams.items()
        if len(agents) >= 2 and ngram_counts[gram] >= 3
    }
    shared = dedup_ngrams(shared)
    results = []
    for gram, agents in sorted(
        shared.items(),
        key=lambda x: ngram_counts[x[0]],
        reverse=True,
    ):
        results.append({
            "phrase": " ".join(gram),
            "count": ngram_counts[gram],
            "agents": sorted(agents),
        })
    return {"cross_agent_phrases": results}


def find_opener_patterns(
    messages: list[dict],
) -> list[dict]:
    """Find repeated message openers per agent."""
    agent_openers: dict[
        str, Counter
    ] = defaultdict(Counter)
    for msg in messages:
        words = msg["content"].lower().split()[:6]
        if len(words) >= 3:
            opener = " ".join(words)
            opener = re.sub(
                r"[^\w\s]", "", opener
            ).strip()
            agent_openers[msg["agent_name"]][
                opener
            ] += 1
    results = []
    for agent, openers in sorted(
        agent_openers.items()
    ):
        for opener, count in openers.most_common():
            if count >= 3:
                results.append({
                    "agent": agent,
                    "opener": opener,
                    "count": count,
                })
    return results


def find_exact_dupes(
    messages: list[dict],
) -> list[dict]:
    """Find exact duplicate messages."""
    seen: dict[str, list[dict]] = defaultdict(list)
    for msg in messages:
        normalized = re.sub(
            r"\s+", " ", msg["content"].strip().lower()
        )
        seen[normalized].append({
            "agent": msg["agent_name"],
            "turn": msg["turn_number"],
        })
    results = []
    for text, entries in seen.items():
        if len(entries) >= 2:
            agents = {e["agent"] for e in entries}
            results.append({
                "text": text[:80],
                "count": len(entries),
                "agents": sorted(agents),
            })
    results.sort(key=lambda x: x["count"], reverse=True)
    return results


def compute_agent_stats(
    con: duckdb.DuckDBPyConnection,
    session_id: str | None = None,
) -> list[dict]:
    """Compute per-agent aggregate statistics."""
    where = ""
    params = []
    if session_id:
        where = "WHERE session_id LIKE ? || '%'"
        params.append(session_id)
    query = (
        "SELECT agent_name, "
        "COUNT(*) AS msg_count, "
        "ROUND(AVG(tokens_completion), 1) "
        "  AS avg_tokens, "
        "ROUND(AVG(generation_time_ms), 0) "
        "  AS avg_gen_ms, "
        "ROUND(AVG(LENGTH(content)), 0) "
        "  AS avg_chars "
        f"FROM messages {where} "
        "GROUP BY agent_name "
        "ORDER BY msg_count DESC"
    )
    rows = con.execute(query, params).fetchall()
    cols = [
        "agent_name", "msg_count", "avg_tokens",
        "avg_gen_ms", "avg_chars",
    ]
    return [dict(zip(cols, r)) for r in rows]


def detect_topics(
    messages: list[dict],
) -> dict[str, object]:
    """Word frequency analysis and stale stretches."""
    all_words: Counter = Counter()
    for msg in messages:
        words = tokenize(msg["content"])
        all_words.update(words)
    stale = _find_stale_stretches(messages)
    return {
        "top_words": all_words.most_common(20),
        "stale_stretches": stale,
    }


def _find_stale_stretches(
    messages: list[dict],
) -> list[dict]:
    """Find stretches where one word dominates 5+ turns.

    Returns stretches with start turn, end turn, and
    the dominating word.
    """
    if len(messages) < 5:
        return []
    window = 5
    stretches = []
    i = 0
    while i <= len(messages) - window:
        chunk = messages[i:i + window]
        words_combined: Counter = Counter()
        for msg in chunk:
            words_combined.update(
                tokenize(msg["content"])
            )
        if not words_combined:
            i += 1
            continue
        top_word, top_count = (
            words_combined.most_common(1)[0]
        )
        if top_count >= window:
            stretches.append({
                "start_turn": chunk[0]["turn_number"],
                "end_turn": chunk[-1]["turn_number"],
                "word": top_word,
                "count": top_count,
            })
            i += window
        else:
            i += 1
    return stretches


def compute_topic_diversity(
    messages: list[dict],
) -> dict[str, float]:
    """Compute unique-word / total-word ratio."""
    all_words = []
    for msg in messages:
        all_words.extend(tokenize(msg["content"]))
    total = len(all_words)
    unique = len(set(all_words))
    ratio = unique / total if total > 0 else 0.0
    return {
        "total_words": total,
        "unique_words": unique,
        "diversity_ratio": round(ratio, 4),
    }


def compute_session_summary(
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Summarize each session with message counts."""
    query = (
        "SELECT s.session_id, s.started_at, "
        "s.ended_at, s.bar_name, s.agent_count, "
        "COUNT(m.message_id) AS msg_count, "
        "COUNT(DISTINCT m.agent_name) "
        "  AS active_agents "
        "FROM sessions s "
        "LEFT JOIN messages m "
        "  ON s.session_id = m.session_id "
        "GROUP BY s.session_id, s.started_at, "
        "  s.ended_at, s.bar_name, s.agent_count "
        "ORDER BY s.started_at"
    )
    rows = con.execute(query).fetchall()
    cols = [
        "session_id", "started_at", "ended_at",
        "bar_name", "agent_count", "msg_count",
        "active_agents",
    ]
    return [dict(zip(cols, r)) for r in rows]


def fetch_regenerations(
    con: duckdb.DuckDBPyConnection,
    session_id: str | None = None,
) -> list[dict]:
    """Query regeneration events."""
    query = (
        "SELECT regen_id, session_id, turn_number, "
        "agent_name, attempt_count, created_at "
        "FROM regenerations"
    )
    params = []
    if session_id:
        query += " WHERE session_id LIKE ? || '%'"
        params.append(session_id)
    query += " ORDER BY created_at"
    try:
        rows = con.execute(query, params).fetchall()
    except duckdb.CatalogException:
        return []
    cols = [
        "regen_id", "session_id", "turn_number",
        "agent_name", "attempt_count", "created_at",
    ]
    return [dict(zip(cols, r)) for r in rows]


def compute_regen_stats(
    regens: list[dict],
) -> dict[str, object]:
    """Compute regeneration statistics."""
    if not regens:
        return {
            "total_regens": 0,
            "by_agent": {},
            "avg_attempts": 0.0,
        }
    by_agent: dict[str, int] = defaultdict(int)
    total_attempts = 0
    for r in regens:
        by_agent[r["agent_name"]] += 1
        total_attempts += r["attempt_count"]
    avg = total_attempts / len(regens) if regens else 0.0
    return {
        "total_regens": len(regens),
        "by_agent": dict(by_agent),
        "avg_attempts": round(avg, 2),
    }


def render_echoes(
    console: Console, data: dict
) -> None:
    """Render echo detection results."""
    console.rule("[bold red]Echo Detection")
    phrases = data.get("cross_agent_phrases", [])
    if phrases:
        table = Table(title="Cross-Agent Phrases")
        table.add_column("Phrase", style="yellow")
        table.add_column("Count", justify="right")
        table.add_column("Agents")
        for p in phrases[:20]:
            table.add_row(
                p["phrase"],
                str(p["count"]),
                ", ".join(p["agents"]),
            )
        console.print(table)
    else:
        console.print("[green]No cross-agent echoes.")


def _render_openers(
    console: Console, openers: list[dict]
) -> None:
    """Render repeated opener patterns."""
    if openers:
        table = Table(title="Repeated Openers")
        table.add_column("Agent", style="cyan")
        table.add_column("Opener", style="yellow")
        table.add_column("Count", justify="right")
        for o in openers[:20]:
            table.add_row(
                o["agent"],
                o["opener"],
                str(o["count"]),
            )
        console.print(table)
    else:
        console.print("[green]No repeated openers.")


def _render_dupes(
    console: Console, dupes: list[dict]
) -> None:
    """Render exact duplicate messages."""
    if dupes:
        table = Table(title="Exact Duplicates")
        table.add_column("Text", style="yellow")
        table.add_column("Count", justify="right")
        table.add_column("Agents")
        for d in dupes[:20]:
            table.add_row(
                d["text"],
                str(d["count"]),
                ", ".join(d["agents"]),
            )
        console.print(table)
    else:
        console.print("[green]No exact duplicates.")


def render_agents(
    console: Console, stats: list[dict]
) -> None:
    """Render per-agent statistics table."""
    console.rule("[bold blue]Agent Statistics")
    if not stats:
        console.print("[dim]No messages found.")
        return
    table = Table(title="Agent Stats")
    table.add_column("Agent", style="cyan")
    table.add_column("Messages", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Gen ms", justify="right")
    table.add_column("Avg Chars", justify="right")
    for s in stats:
        table.add_row(
            s["agent_name"],
            str(s["msg_count"]),
            str(s["avg_tokens"]),
            str(s["avg_gen_ms"]),
            str(s["avg_chars"]),
        )
    console.print(table)


def render_topics(
    console: Console,
    data: dict,
    diversity: dict,
) -> None:
    """Render topic analysis results."""
    console.rule("[bold green]Topic Analysis")
    console.print(
        f"Vocabulary: {diversity['unique_words']} "
        f"unique / {diversity['total_words']} total "
        f"(ratio: {diversity['diversity_ratio']})"
    )
    top = data.get("top_words", [])
    if top:
        table = Table(title="Top Words")
        table.add_column("Word", style="cyan")
        table.add_column("Count", justify="right")
        for word, count in top:
            table.add_row(word, str(count))
        console.print(table)
    stale = data.get("stale_stretches", [])
    if stale:
        st = Table(title="Stale Stretches")
        st.add_column("Turns", style="yellow")
        st.add_column("Word", style="red")
        st.add_column("Count", justify="right")
        for s in stale[:15]:
            st.add_row(
                f"{s['start_turn']}-{s['end_turn']}",
                s["word"],
                str(s["count"]),
            )
        console.print(st)


def render_sessions(
    console: Console, summaries: list[dict]
) -> None:
    """Render session overview table."""
    console.rule("[bold magenta]Sessions")
    if not summaries:
        console.print("[dim]No sessions found.")
        return
    table = Table(title="Session Overview")
    table.add_column("ID (short)", style="cyan")
    table.add_column("Started")
    table.add_column("Bar")
    table.add_column("Agents", justify="right")
    table.add_column("Messages", justify="right")
    for s in summaries:
        sid = str(s["session_id"])[:8]
        started = str(s["started_at"] or "")[:19]
        table.add_row(
            sid,
            started,
            s["bar_name"] or "",
            str(s["agent_count"] or 0),
            str(s["msg_count"]),
        )
    console.print(table)


def render_regens(
    console: Console, stats: dict, regens: list[dict]
) -> None:
    """Render regeneration statistics."""
    console.rule("[bold yellow]Regenerations")
    if stats["total_regens"] == 0:
        console.print("[green]No regenerations recorded.")
        return
    console.print(
        f"Total: {stats['total_regens']} regens, "
        f"avg {stats['avg_attempts']} attempts each"
    )
    by_agent = stats.get("by_agent", {})
    if by_agent:
        table = Table(title="Regenerations by Agent")
        table.add_column("Agent", style="cyan")
        table.add_column("Count", justify="right")
        for agent, count in sorted(
            by_agent.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            table.add_row(agent, str(count))
        console.print(table)
    if regens:
        recent = Table(title="Recent Regenerations")
        recent.add_column("Turn", justify="right")
        recent.add_column("Agent", style="cyan")
        recent.add_column("Attempts", justify="right")
        for r in regens[-10:]:
            recent.add_row(
                str(r["turn_number"]),
                r["agent_name"],
                str(r["attempt_count"]),
            )
        console.print(recent)


def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Dive Bar conversation analyzer",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help="Path to DuckDB file",
    )
    parser.add_argument(
        "--session", default=None,
        help="Filter to a specific session ID",
    )
    sub = parser.add_subparsers(dest="command")
    for name, hlp in [
        ("echoes", "Repetition detection"),
        ("agents", "Per-agent statistics"),
        ("topics", "Topic diversity analysis"),
        ("sessions", "Session overview"),
        ("regens", "Regeneration statistics"),
        ("all", "Full report"),
    ]:
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument(
            "--session", default=None,
            help="Filter to a session ID (prefix)",
        )
    return parser


def main() -> None:
    """Entry point — dispatch to subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    console = Console()
    con = open_db(args.db)
    try:
        _dispatch(console, con, args)
    finally:
        con.close()


def _dispatch(
    console: Console,
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> None:
    """Route subcommand to the right handler."""
    cmd = args.command
    sid = getattr(args, "session", None)
    if cmd in ("echoes", "all"):
        msgs = fetch_messages(con, sid)
        echoes = detect_echoes(msgs)
        openers = find_opener_patterns(msgs)
        dupes = find_exact_dupes(msgs)
        render_echoes(console, echoes)
        _render_openers(console, openers)
        _render_dupes(console, dupes)
    if cmd in ("agents", "all"):
        stats = compute_agent_stats(con, sid)
        render_agents(console, stats)
    if cmd in ("topics", "all"):
        msgs = fetch_messages(con, sid)
        topic_data = detect_topics(msgs)
        diversity = compute_topic_diversity(msgs)
        render_topics(console, topic_data, diversity)
    if cmd in ("sessions", "all"):
        summaries = compute_session_summary(con)
        render_sessions(console, summaries)
    if cmd in ("regens", "all"):
        regens = fetch_regenerations(con, sid)
        stats = compute_regen_stats(regens)
        render_regens(console, stats, regens)


if __name__ == "__main__":
    main()
