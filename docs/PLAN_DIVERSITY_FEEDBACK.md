# Diversity Feedback System Implementation Plan

## Overview

Real-time diversity feedback system for the Dive Bar
conversation simulation. Detects repetitive patterns in
generated responses and triggers regeneration when
diversity falls below acceptable thresholds.

## Architecture

```
dive_bar/
  diversity.py          NEW - Core diversity scoring
  pattern_store.py      NEW - Persistent pattern learning
  app.py                MODIFY - Regeneration loop
  agent.py              MODIFY - Dynamic suppression
  db.py                 MODIFY - Logging tables
  models.py             MODIFY - New dataclasses
```

## 1. Diversity Index Function

### File: dive_bar/diversity.py

```python
def compute_diversity_score(
    response: str,
    history: list[Message],
    window: int = 10,
) -> DiversityResult:
    """Check response against recent history.

    Returns DiversityResult with score (0-1) and problems.
    """
```

### Detection Components

**1.1 N-gram Overlap Detection**
- Extract 3-6 word n-grams from response
- Compare against n-grams from recent history
- Flag if >30% of response n-grams appear in history
- Return the specific repeated phrases

**1.2 Formulaic Opener Detection**
- Check first 6 words against known patterns
- Track per-agent opener frequency
- Flag if opener used >2 times in window

**1.3 Structural Similarity Detection**
- Compare sentence count, question marks, commas
- Check against last 3 responses from same agent
- Return similarity score 0-1

**1.4 Score Aggregation**

```python
@dataclass
class DiversityResult:
    score: float              # 0.0 (bad) to 1.0 (good)
    passed: bool              # score >= threshold
    problems: list[str]       # Human-readable problems
    repeated_ngrams: list[str]
    formulaic_opener: str | None
    structural_score: float

# Weights
W_NGRAM = 0.50
W_OPENER = 0.25
W_STRUCT = 0.25
```

### Performance Target
- compute_diversity_score: <50ms
- Use set operations for O(1) n-gram lookup
- Pre-tokenize history once per turn

---

## 2. Regeneration Loop

### File: dive_bar/app.py

Modify `_generate_turn()`:

```python
MAX_REGEN_ATTEMPTS = 3
DIVERSITY_THRESHOLD = 0.6

def _generate_turn(self, name: str) -> dict:
    # ... existing generation code ...

    # NEW: Diversity check loop
    attempt = 0
    regen_reasons = []
    while attempt < MAX_REGEN_ATTEMPTS:
        diversity = compute_diversity_score(
            content, self.history, window=10
        )
        if diversity.passed:
            break

        regen_reasons.append(diversity.problems)
        feedback = self._build_rephrase_prompt(
            name, content, diversity
        )
        result = self.engine.generate(feedback, stop=stop)
        content = self._clean_response(name, result.content)
        attempt += 1

    if attempt > 0:
        self._log_regeneration(name, attempt, regen_reasons)
```

### Rephrase Prompt

```python
def _build_rephrase_prompt(
    self, name: str, original: str, diversity: DiversityResult
) -> list[dict]:
    problems = "\n".join(f"- {p}" for p in diversity.problems[:3])
    return [
        {"role": "system", "content": self.agents[name].system_prompt},
        {"role": "user", "content": (
            f"You just said: \"{original}\"\n\n"
            f"Problems:\n{problems}\n\n"
            f"Say the same thing completely differently. "
            f"Fresh phrasing. 1-2 sentences."
        )},
    ]
```

---

## 3. Real-Time Prompt Adaptation

### File: dive_bar/pattern_store.py

```python
@dataclass
class SuppressedPattern:
    pattern: str
    pattern_type: str      # "ngram", "opener", "word"
    hit_count: int
    first_seen: float
    last_seen: float
    agents: set[str]


class PatternStore:
    def __init__(self, db_path: str):
        self._patterns: dict[str, SuppressedPattern] = {}
        self._load_from_db()

    def record_pattern(
        self, pattern: str, pattern_type: str, agent_name: str
    ) -> None:
        """Record pattern that triggered regeneration."""

    def get_suppressions(
        self, agent_name: str, limit: int = 5
    ) -> list[str]:
        """Get top patterns to suppress for this agent."""
```

### File: dive_bar/agent.py

Modify to include dynamic suppressions:

```python
def _build_system(self) -> str:
    base = SYSTEM_TEMPLATE.format(...)

    if self._pattern_store:
        suppressions = self._pattern_store.get_suppressions(
            self.config.name, limit=5
        )
        if suppressions:
            avoid_list = ", ".join(f'"{s}"' for s in suppressions)
            base += f"\n\nAvoid: {avoid_list}."
    return base

def refresh_suppressions(self) -> None:
    """Rebuild system prompt with updated patterns."""
    self.system_prompt = self._build_system()
```

Refresh every 20 turns (configurable).

---

## 4. Database Logging

### File: dive_bar/db.py

New tables:

```sql
CREATE TABLE IF NOT EXISTS regenerations (
    regen_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    turn_number INTEGER,
    agent_name VARCHAR NOT NULL,
    attempt_number INTEGER NOT NULL,
    original_content TEXT,
    final_content TEXT,
    problems TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
)

CREATE TABLE IF NOT EXISTS suppressed_patterns (
    pattern_id VARCHAR PRIMARY KEY,
    pattern TEXT NOT NULL,
    pattern_type VARCHAR NOT NULL,
    hit_count INTEGER DEFAULT 1,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    agents TEXT,
    UNIQUE(pattern, pattern_type)
)
```

### Analysis (analyze.py)

New subcommands:
- `python analyze.py regens` — regeneration stats
- `python analyze.py patterns` — suppressed pattern stats

---

## 5. Configuration

### config.toml additions

```toml
[diversity]
enabled = true
threshold = 0.6
max_retries = 3
window_size = 10
ngram_range = [3, 6]
refresh_interval = 20
```

### models.py additions

```python
@dataclass
class DiversityConfig:
    enabled: bool = True
    threshold: float = 0.6
    max_retries: int = 3
    window_size: int = 10
    ngram_min: int = 3
    ngram_max: int = 6
    refresh_interval: int = 20
```

---

## 6. Implementation Order

### Phase 1: Core Detection
1. Create dive_bar/diversity.py
2. Copy tokenize/extract_ngrams from analyze.py
3. Implement detection functions
4. Add DiversityResult to models.py

### Phase 2: Regeneration Loop
1. Add constants to app.py
2. Modify _generate_turn() with diversity check
3. Implement _build_rephrase_prompt()
4. Add status updates during regen

### Phase 3: Database Logging
1. Add tables to db.py
2. Implement log_regeneration()
3. Implement upsert_pattern()

### Phase 4: Pattern Store
1. Create pattern_store.py
2. Implement PatternStore class
3. Integrate with app.py

### Phase 5: Prompt Adaptation
1. Modify Agent to accept PatternStore
2. Modify _build_system() for suppressions
3. Add refresh trigger in app.py

### Phase 6: Analysis Tools
1. Add regens subcommand
2. Add patterns subcommand

---

## 7. Performance Budget

| Operation                | Target    |
|--------------------------|-----------|
| compute_diversity_score  | <50ms     |
| Regeneration (1 attempt) | +500ms    |
| Pattern DB lookup        | <5ms      |
| Prompt refresh           | <10ms     |

Total overhead per turn (no regen): <100ms

---

## 8. Rollback

Set `[diversity] enabled = false` in config.toml.
All new code is additive — existing behavior preserved.
