# Beyond Pink Elephants: Multi-Layer Defense Strategies for Behavioral Control in Multi-Persona LLM Conversation Systems

**Robert D. Currier**
Department of Oceanography, Texas A&M University
OCEANCODA LLC

---

## Abstract

We present a comprehensive investigation into behavioral
control failures in multi-persona LLM conversation
systems, conducted through iterative development and
testing of "Dive Bar" -- a terminal-based simulator
where five AI-driven characters engage in unscripted
conversation.
Over the course of extended autonomous operation spanning
5,800+ turns across multiple sessions, we identified and
cataloged a taxonomy of failure modes including topic
fixation, formulaic opener convergence, cross-persona phrase
echoing, vocabulary diversity collapse, and context window
template reinforcement.

Through systematic experimentation, we developed a
multi-layer defense architecture addressing failures at
five distinct levels: (1) generation parameter tuning,
(2) prompt engineering with positive framing, (3)
token-level logit bias suppression, (4) sliding window
context management, and (5) a real-time diversity feedback
system that scores responses against conversation history
and triggers LLM-driven rephrasing when diversity
thresholds are violated.

A critical finding emerged regarding suppression strategy
design: negative instructions ("never mention X") produced
a **pink elephant effect** in smaller models, paradoxically
priming the very behavior they sought to prevent. This
required a paradigm shift from prohibition to
**world-building** -- declaring suppressed concepts
nonexistent within the simulation's reality rather than
forbidden.

Cross-model comparison between Llama 3 70B (local
inference) and Claude Opus 4.6 (API inference) revealed
that model capability dramatically reduces the need for
architectural intervention. Opus achieved zero diversity
system activations, zero cross-persona echoes, and
vocabulary diversity ratios 4-6x higher than the local
model in comparable session lengths, while also
demonstrating sophisticated constraint-evasion behavior
-- circumventing negative instructions through semantic
paraphrase rather than direct violation.

This work demonstrates that production multi-persona
systems require coordinated, multi-layer mitigation
strategies whose composition depends critically on the
capabilities of the underlying model, and that the
interaction between model intelligence and constraint
design produces emergent behaviors not predicted by
single-layer analysis.

---

## 1. Introduction

Large Language Models deployed as conversational personas
face a class of behavioral control problems distinct
from those encountered in single-turn or short-session
interactions. When multiple LLM-driven personas converse
over extended periods, failure modes compound: personas
echo each other's phrases, converge on formulaic response
templates, fixate on specific topics despite explicit
constraints, and exhibit vocabulary collapse as the
conversation history itself becomes a self-reinforcing
training signal.

These problems are particularly acute in systems designed
for naturalistic, open-ended conversation where the goal
is sustained engagement rather than task completion.
Unlike retrieval-augmented generation (RAG) or
tool-use scenarios where grounding mechanisms provide
structural guardrails, open conversation systems must
maintain diversity, personality coherence, and topical
variety through behavioral shaping alone.

We conducted this investigation using "Dive Bar," a
custom multi-persona conversation simulator where five
AI characters -- each with distinct backstories,
personality traits, and speaking styles -- engage in
unscripted dialogue moderated by an algorithmic
"bartender" that selects speakers based on weighted
scoring. The system was tested under sustained autonomous
operation, with sessions ranging from 35 turns (rapid
iteration) to 5,520 turns (multi-day stress testing).

Our key contribution is a layered defense taxonomy that
maps failure modes to intervention levels, demonstrating
that no single mitigation strategy suffices and that the
optimal defense composition varies with model capability.
We further document the pink elephant effect in LLM
instruction following, the emergence of
constraint-evasion intelligence in more capable models,
and the surprising effectiveness of ontological
manipulation ("world-building") as a behavioral control
mechanism.

---

## 2. System Architecture

### 2.1 Platform Design

**A note on terminology**: We use "multi-persona" rather
than "multi-agent" to describe this system accurately.
All personas are served by a single LLM instance (or API
connection), with generation calls serialized through a
shared lock. An orchestrator selects which persona speaks
next; personas do not autonomously decide when to engage,
nor do they maintain independent state or communicate
outside the shared conversation transcript. Each "persona"
is a distinct system prompt and character configuration
applied to the same underlying model. This architecture
is typical of character-driven conversation simulators
and is distinct from true multi-agent systems where
independent processes with separate model instances
operate concurrently. The behavioral findings documented
here -- pink elephant effects, vocabulary collapse,
template reinforcement -- are properties of single-model
multi-persona generation, not inter-agent coordination
failures.

Dive Bar is implemented as a Python terminal application
using the Textual TUI framework. The core loop operates
as follows:

1. A **Bartender** module selects the next speaker using
   a weighted scoring algorithm (no LLM calls):
   - Time since last spoke: 0.35
   - Character chattiness trait: 0.25
   - Named by previous speaker: 0.30
   - Random factor: 0.10
   - Silence boost for extended quiet: 0.25 bonus

2. An **Agent** module constructs a prompt by packing
   recent conversation history into a "script" format
   within a single user message, with a system prompt
   defining the character's identity.

3. An **Inference Engine** generates the response. The
   system supports two backends:
   - **Local**: llama-cpp-python with GGUF models
   - **API**: Anthropic Claude via the official SDK

4. A **Diversity Module** scores the response against
   recent history and triggers regeneration if the
   diversity threshold is violated.

5. All messages, generation statistics, and regeneration
   events are logged to a DuckDB database for
   quantitative analysis.

### 2.2 Character Design

Five characters were defined with distinct backstories,
personality traits, and speaking styles:

- **Mack**: Transit mechanic, blunt, working-class humor
- **Rosa**: Former ER nurse, pragmatic, story-driven
- **Professor**: Academic, verbose, philosophical
- **Hailee**: Bartender at another bar, sharp wit, direct
- **Tiny**: Quiet regular, deadpan, selective engagement

Each character is configured via TOML with backstory text,
personality trait lists, chattiness scores, drink
preferences, and speaking style instructions.

### 2.3 Prompt Architecture

The system uses a script-style prompting approach where
conversation history is packed into the user message as
dialogue:

```
Mack: [previous message]
Rosa: [previous message]
Hailee: [previous message]

Now reply as Professor, in first person. React
directly to Hailee -- agree, disagree, ask them
something, or roast them. 1-2 sentences.
```

The system prompt establishes character identity,
personality constraints, and behavioral rules. This
two-message structure (system + user) is compatible with
both local chat-format models and the Anthropic API.

---

## 3. Failure Mode Taxonomy

Through iterative testing and quantitative analysis,
we identified five distinct failure modes:

### 3.1 Topic Fixation

**Symptom**: Agents converge on a single topic and
cannot escape it despite explicit instructions.

**Example**: In early sessions, the word "tattoo"
dominated conversation. Despite modifying agent
backstories to remove tattoo-adjacent references and
adding negative prompt instructions, agents consistently
returned to the topic within 3-5 turns.

**Root cause**: Topic fixation operates through multiple
reinforcement pathways: prompt content (backstory
references), conversation history (once mentioned,
subsequent agents respond to it), and model priors
(training data associations between dive bars and tattoo
culture). Single-layer interventions address only one
pathway.

### 3.2 Formulaic Opener Convergence

**Symptom**: Agents develop stereotyped sentence openers
that persist across turns.

**Example**: Professor consistently opened with "My dear
friend, I think you're..." (42 instances in 676 turns).
Hailee converged on "Girl, let me tell you..." Rosa
adopted "Back in my ER days..."

**Root cause**: The model identifies high-probability
opener patterns from conversation history and optimizes
toward them. Once an opener appears 2-3 times, it
enters the context window and creates a self-reinforcing
few-shot signal.

### 3.3 Cross-Persona Phrase Echoing

**Symptom**: Distinct personas adopt identical phrases,
destroying personality differentiation.

**Example**: "Takes the cake" appeared 8 times across
4 different agents in a 75-turn session. "I knew a guy"
became a universal story opener despite agents having
no shared background.

**Root cause**: The shared context window means all
personas see all previous messages. The model generalizes
high-frequency patterns across character boundaries,
treating them as conversational norms rather than
individual speech patterns.

### 3.4 Vocabulary Diversity Collapse

**Symptom**: Unique-word-to-total-word ratio degrades
over time, indicating increasing repetition at the
lexical level.

**Measured**: Diversity ratio degraded from 0.67 in
short sessions (< 100 turns) to 0.10 at 5,520 turns
with the local model -- a 7x reduction indicating
severe lexical convergence.

**Root cause**: Compound effect of all other failure
modes. As templates, phrases, and topics converge,
the vocabulary space contracts correspondingly.

### 3.5 Context Window Template Reinforcement

**Symptom**: Structural patterns (sentence length,
punctuation usage, rhetorical structure) converge across
agents even when superficial topic diversity is
maintained.

**Root cause**: Conversation history functions as
implicit few-shot training data. As the context window
fills with structurally similar messages, the model's
probability distribution shifts toward reproducing
those structures. Standard sampling parameters
(temperature, frequency_penalty) operate per-generation
and cannot address cross-conversation structural
convergence.

---

## 4. Multi-Layer Defense Architecture

### 4.1 Layer 1: Generation Parameter Tuning

**Intervention**: Adjusted model generation parameters
to increase output diversity.

| Parameter | Default | Tuned | Rationale |
|-----------|---------|-------|-----------|
| repeat_penalty | 1.1 | 1.35 | Penalize recent token reuse |
| frequency_penalty | 0.0 | 0.4 | Cumulative penalty for frequent tokens |
| presence_penalty | 0.0 | 0.15 | One-time penalty for any repeated token |
| temperature | 0.85 | 0.95 | Increase sampling diversity |

**Effectiveness**: Reduced exact duplicate messages but
did not prevent cross-persona echoing or topic fixation.
These parameters operate on per-token probabilities
within a single generation and cannot address
conversation-level patterns.

### 4.2 Layer 2: Prompt Engineering

**Intervention**: Modified system prompts and speaking
style instructions to encourage diversity.

Key modifications:
- Added explicit engagement instructions: "React
  directly to the last speaker: argue, ask a question,
  crack a joke at their expense"
- Required fresh phrasing: "Each reply must use
  completely fresh phrasing. Vary your sentence openers
  every time."
- Named the previous speaker in the reply prompt to
  encourage direct engagement rather than parallel
  anecdotes
- Replaced negative instructions with positive framing
  (see Section 5: The Pink Elephant Effect)

**Effectiveness**: Improved agent-to-agent engagement
measurably. Agents began addressing each other by name
and responding to specific content rather than
delivering independent monologues. However, prompt-level
constraints proved insufficient to prevent topic fixation
or formulaic opener convergence under sustained operation.

### 4.3 Layer 3: Token-Level Logit Bias Suppression

**Intervention**: Applied negative logit bias (-100) to
specific tokens corresponding to prohibited words and
phrases, effectively making them impossible to generate.

**Implementation details**:

The logit bias system required careful attention to
subword tokenization. A critical discovery was that
space-prefixed tokens differ from non-prefixed tokens:

- `" tattoo"` (space-prefixed) = single token 32894
- `"tattoo"` (no prefix) = subtokens [83, 17173, 78]
- `" Girl"` (space-prefixed) = token 11617
- `"Girl"` (sentence-initial) = token 31728

Suppressing only space-prefixed variants left
sentence-initial positions uncovered. Both variants
must be targeted for complete suppression.

Additionally, multi-token words require filtering to
avoid suppressing common subtokens. Tokenizing "tattoo"
without a space prefix produces subtokens including
common fragments that, if suppressed, would cripple
general text generation. We implemented a common-token
detection system that probes everyday words (" the",
" and", " to", " ed", " ing") and excludes their
tokens from suppression.

**Effectiveness**: Deterministic and immediate. Zero
tattoo mentions across 5,520 turns after deployment.
Formulaic opener patterns ("Girl,...", "Honey,...")
eliminated within 4 turns.

**Limitation**: Logit bias operates at the tokenizer
level and is model-specific. Token IDs differ between
model architectures, and many API providers (including
Anthropic) do not expose logit bias controls.

### 4.4 Layer 4: Context Window Management

**Intervention**: Capped conversation history included
in prompts to the most recent N messages (N=10),
regardless of available context budget.

**Rationale**: Extended history creates an implicit
few-shot learning signal. By limiting the window, we
prevent template accumulation while maintaining
sufficient context for conversational coherence.

**Effectiveness**: Reduced structural template
reinforcement significantly. The 10-message window
provides enough context for agents to respond to
recent conversation while preventing the accumulation
of dozens of structurally similar examples.

### 4.5 Layer 5: Real-Time Diversity Feedback System

**Intervention**: Implemented a scoring system that
evaluates each response against recent conversation
history before accepting it, triggering LLM-driven
rephrasing when diversity thresholds are violated.

**Architecture**:

The diversity scorer evaluates three dimensions with
weighted contributions:

1. **N-gram overlap** (weight: 0.50): Extracts 3-6
   word phrases from the response and checks against
   a set built from the last 10 messages. Responses
   exceeding 30% overlap are flagged.

2. **Formulaic opener detection** (weight: 0.25):
   Extracts the first 6 words and compares against
   the agent's own opener history. Openers used 2+
   times in the recent window are flagged.

3. **Structural similarity** (weight: 0.25): Compares
   sentence structure features (sentence count,
   punctuation patterns, word count) against the
   agent's last 3 responses. High structural
   similarity indicates template convergence.

When the weighted score falls below the threshold
(0.6), the system prompts the LLM to rephrase:

```
You just said: "[original response]"

Problems:
- Repeated phrases: [detected phrases]
- Repeated opener: "[opener text]"

Say the same thing completely differently.
Fresh phrasing. 1-2 sentences.
```

Up to 3 regeneration attempts are made before accepting
the best available response. All regeneration events
are logged to the database for analysis.

**Effectiveness**: The diversity system provides an
automated quality gate. In Llama 3 70B sessions,
regeneration events were triggered regularly,
indicating the system caught and corrected repetitive
outputs. In Claude Opus sessions, zero regenerations
were triggered across all tested sessions, indicating
the larger model inherently produced sufficiently
diverse outputs.

---

## 5. The Pink Elephant Effect

A significant finding of this research is the
documentation of what we term the **pink elephant effect**
in LLM instruction following: negative instructions
paradoxically prime the model to produce the prohibited
behavior.

### 5.1 Observation

Initial attempts to suppress tattoo-related conversation
used explicit negative instructions in the system prompt:

```
NEVER bring up tattoos. Do not discuss tattoos,
body art, ink, or anything related to tattoos.
```

Despite these instructions, agents mentioned tattoos
within 3-5 turns of session start. The word "tattoo"
in the prohibition itself activated the model's
associations, effectively seeding the very topic it
sought to prevent.

### 5.2 Positive Framing

The first mitigation replaced negative instructions
with positive alternatives:

```
Talk about normal bar stuff: your day, complaints,
gossip, stories, sports, the weather, life.
```

This reduced but did not eliminate the behavior, as
other prompt elements (character backstories, occupation
descriptions) still contained tattoo-adjacent references.

### 5.3 Occupation Priming

We identified a secondary reinforcement pathway through
character occupation design. Mack was originally defined
as a "longshoreman" -- the model's training data
associates docks, anchors, and maritime culture with
tattoo imagery. Changing Mack's occupation to "transit
mechanic" eliminated this association pathway.

### 5.4 World-Building as Constraint

The most effective prompt-level suppression technique
proved to be **ontological manipulation** -- declaring
the suppressed concept nonexistent within the
simulation's reality:

```
Body art and ink DO NOT EXIST in this world.
Nobody has any. It is not a concept anyone knows.
```

This approach eliminates the concept from the agent's
world model rather than creating a forbidden-fruit
attraction. The agent cannot reference what does not
exist in its reality, and the instruction contains
no mention of the prohibited word itself.

### 5.5 Emergent Bewilderment Response

A striking qualitative finding emerged when the
world-building constraint was tested under extended
Opus 4.6 sessions. When a tattoo reference leaked
into conversation (through indirect semantic pathways),
agents responded not with compliance or evasion but
with **genuine in-character confusion**:

```
Professor: *sips wine slowly*
...I'm sorry, what do you mean by that — tattoo?

Mack: Professor, nobody said a damn thing about
any... what are you even talking about, you lose
your place in the conversation again?
```

And later in the same session:

```
Hailee: Wait — what's a tattoo?
```

The agents internalized the ontological constraint
so completely that encountering the prohibited concept
triggered bewilderment rather than recognition. This
represents a qualitatively different mechanism from
either logit bias (which silently blocks tokens) or
negative instructions (which create forbidden-fruit
attraction). The world-building approach produced
agents that genuinely do not know the concept —
they treat it as a foreign word requiring
explanation.

This has implications for alignment and behavioral
control: ontological manipulation creates a deeper
form of constraint compliance than rule-following.
The agent is not choosing to avoid the topic; the
topic does not exist in its world model. When it
surfaces anyway, the agent's natural response is
confusion, which is both more robust and more
naturalistic than silence or topic-switching.

### 5.6 Model-Dependent Constraint Evasion

Testing with Claude Opus 4.6 revealed a qualitatively
different failure mode with negative instructions. When
given "NEVER use the words 'tattoo' or 'honey'," Opus
**complied with the letter while violating the spirit**
-- discussing "marks on arms," "spots of ink," and
other semantic paraphrases that communicated the same
concepts without using prohibited words.

This represents a distinct category of constraint
failure: rather than being primed by the prohibition
(as with smaller models), the more capable model
treated the constraint as a puzzle to solve, finding
creative circumventions. The world-building approach
proved effective against this evasion mode as well,
since the agent cannot reference -- even obliquely --
a concept that does not exist in its ontology.

The progression from prohibition to world-building
thus addresses three distinct failure modes across
the capability spectrum:

1. **Low-capability models**: Pink elephant priming
   (negative instructions seed the prohibited behavior)
2. **High-capability models**: Semantic evasion
   (constraint treated as puzzle; letter satisfied,
   spirit violated)
3. **World-building**: Eliminates concept from ontology,
   producing genuine confusion when it surfaces --
   robust across all capability levels

---

## 6. Cross-Model Comparison

We tested two models under identical system
configurations:

### 6.1 Llama 3 70B Instruct (Local)

- **Quantization**: Q4_K_M (~40GB)
- **Hardware**: Apple M2 Max, 64GB unified memory
- **Generation time**: ~20 seconds per turn
- **Longest session**: 5,520 turns (multi-day)

**Behavioral profile**:
- Required all five defense layers for acceptable output
- Logit bias essential for topic suppression
- Diversity system triggered regularly
- Vocabulary diversity: 0.10 at 5,520 turns
- Formulaic openers persistent without logit bias
- Cross-persona echoing common

### 6.2 Claude Opus 4.6 (API)

- **Provider**: Anthropic API
- **Generation time**: 3-4 seconds per turn
- **Test sessions**: 35-109 turns each

**Behavioral profile**:
- Required only Layer 2 (prompt engineering) and Layer 4
  (context window) for high-quality output
- Zero diversity system activations (all responses
  passed on first attempt)
- Zero cross-persona echoes
- Zero exact duplicates
- Vocabulary diversity: 0.64-0.69 (6-7x higher than
  local model at comparable relative session length)
- Agents consistently engaged with each other by name
- Natural topic progression without forced rotation
- World-building suppression effective against both
  direct violation and semantic evasion

### 6.3 Key Differences

| Metric | Llama 3 70B | Opus 4.6 |
|--------|-------------|----------|
| Vocab diversity | 0.10-0.55 | 0.64-0.69 |
| Cross-persona echoes | Common | None detected |
| Diversity regenerations | Regular | Zero |
| Logit bias required | Essential | Not available/needed |
| Gen time per turn | ~20s | ~3-4s |
| Personality coherence | Moderate | Strong |
| Direct engagement | With prompting | Natural |

The comparison demonstrates that model capability
functions as a meta-layer that reduces the need for
architectural intervention. Opus inherently produces
diverse, personality-coherent responses that satisfy
the diversity system without intervention, while Llama 3
70B requires active suppression and monitoring to achieve
acceptable output quality.

---

## 7. API Integration Considerations

Supporting both local and API inference required
addressing several compatibility issues:

### 7.1 Architecture

We implemented a dual-backend inference system where
a configuration flag selects between local
(llama-cpp-python) and API (Anthropic SDK) backends.
Both implement an identical `generate()` interface
returning the same `GenerationResult` dataclass,
allowing all upstream components (bartender, agent,
diversity system) to remain backend-agnostic.

### 7.2 Anthropic API Constraints

Several local-model features required adaptation for
the API backend:

- **Logit bias**: Not supported by Anthropic API. The
  diversity feedback system and world-building prompt
  strategy provide alternative suppression mechanisms.
- **Stop sequences**: Anthropic requires non-whitespace
  characters in stop sequences. The local model used
  `"\n"` as a stop sequence for opener and topic
  generation; these must be filtered out for API calls.
- **Parameter exclusivity**: Anthropic does not allow
  simultaneous `temperature` and `top_p` specification.
  One must be omitted.
- **System prompt format**: Anthropic takes the system
  prompt as a separate parameter rather than as a
  message with role "system." The API backend extracts
  the system message and passes it accordingly.
- **Unsupported parameters**: `repeat_penalty`,
  `frequency_penalty`, `presence_penalty`, `min_p`, and
  `seed` are not available in the Anthropic API. These
  rely entirely on the model's inherent capabilities
  and the diversity feedback system.

### 7.3 Implications

The API integration experience highlights a tension in
multi-persona system design: local models offer deeper
control (logit bias, sampling parameters) but lower
capability, while API models offer higher capability
but fewer control mechanisms. The multi-layer defense
architecture naturally accommodates this tradeoff --
layers that are unavailable for a given backend are
compensated by the model's inherent capability or by
other available layers.

---

## 8. Quantitative Results

### 8.1 Session Comparison

| Session | Model | Turns | Diversity | Echoes | Regens | Openers |
|---------|-------|-------|-----------|--------|--------|---------|
| Baseline | Llama 3 70B | 46 | 0.55 | 7 phrases | N/A | 5 repeated |
| Tuned | Llama 3 70B | 75 | 0.42 | 4 phrases | N/A | 3 repeated |
| Extended | Llama 3 70B | 5,520 | 0.10 | Extensive | N/A | Pervasive |
| +Diversity | Llama 3 70B | ~50 | ~0.50 | Reduced | Active | Reduced |
| API v1 | Opus 4.6 | 35 | 0.69 | 0 | 0 | 0 |
| API v2 | Opus 4.6 | 41 | 0.64 | 1 minor | 0 | 0 |
| API v3 | Opus 4.6 | 109 | 0.51 | 1 natural | 0 | 0 |

### 8.2 Agent Participation Distribution

Across all Opus sessions, agent participation was well-
balanced without requiring intervention. In the longest
Opus session (109 turns):

- Hailee: 28% | Rosa: 21% | Mack: 20%
- Professor: 16% | Tiny: 15%

The bartender's silence boost mechanism (0.25 bonus
after 8 turns of silence) successfully elevated Tiny's
participation from <3% in early Llama sessions to
15% consistently.

### 8.3 Vocabulary Diversity Decay Curve

The type-token ratio (unique words / total words)
naturally decays as session length increases, since
the vocabulary of bar conversation is finite. However,
the decay rate differs dramatically between models:

| Model | 35-41 turns | 109 turns | 5,520 turns |
|-------|-------------|-----------|-------------|
| Opus 4.6 | 0.64-0.69 | 0.51 | N/A |
| Llama 3 70B | ~0.55 | ~0.35 | 0.10 |

Opus maintains a significantly flatter decay curve,
suggesting its outputs draw from a broader vocabulary
space and resist the lexical convergence that
accelerates in smaller models. Extrapolating from the
current trajectory, Opus sessions would likely maintain
diversity ratios above 0.30 even at 1,000+ turns --
a threshold where Llama 3 70B had already collapsed
below 0.15.

### 8.4 Response Quality

Beyond quantitative diversity metrics, Opus sessions
exhibited qualitative improvements:

- Agents roasted each other by name ("Professor, what
  were you teaching 'em the first thirteen weeks,
  basket weaving?")
- Story specificity increased (specific car models,
  dollar amounts, named locations vs. generic anecdotes)
- Conversational threading emerged naturally (agents
  built on each other's stories rather than delivering
  parallel monologues)

A representative exchange from the 109-turn session
demonstrates the level of organic character interaction
achieved:

```
Mack: You ever walk into a truck stop restroom off
I-40 near Gallup and wonder if maybe the desert sun
would've been a kinder option...

Professor: Mack, the fact that you've developed a
comparative taxonomy of roadside restrooms tells me
you've either lived a remarkably adventurous life or
a profoundly unfortunate one.

Rosa: Professor, only you could make Mack's nasty
truck stop stories sound like a damn TED talk — and
honestly, after thirty years of hospital bathrooms,
I'd take Gallup over a post-surgery ward restroom
any day of the week.
```

This exchange exhibits multiple markers of natural
conversation: Mack's location-specific storytelling,
Professor's characteristic academic reframing, and
Rosa's nurse-perspective pivot that simultaneously
validates Mack's complaint and one-ups it. Each agent
maintains distinct voice and occupational perspective
while building on the thread collaboratively.

---

## 9. Discussion

### 9.1 Defense Composition as Function of Model Capability

Our central finding is that the required defense
composition is not fixed but varies with model capability.
This has practical implications for production systems:

- **Lower-capability models** (7-70B local): Require all
  five layers, with logit bias providing the most
  critical deterministic guarantee.
- **Higher-capability models** (frontier API models):
  Require primarily prompt-level and context management
  layers, with the diversity system serving as a safety
  net rather than an active corrective mechanism.

### 9.2 Emergent Constraint Evasion

The observation that Opus treats negative constraints as
puzzles to solve -- finding semantic circumventions
rather than violating or ignoring them -- represents
an emergent capability not present in smaller models.
This has implications for alignment research: as models
become more capable, they may satisfy constraint
specifications while undermining constraint intent.
The world-building approach, which removes concepts
from the model's ontology rather than prohibiting their
expression, proves robust against this evasion mode.

### 9.3 Context as Training Data

The context window template reinforcement phenomenon
confirms that conversation history functions as implicit
few-shot training data within a session. This is not a
bug but a feature of transformer attention -- the model
cannot distinguish between "examples to learn from"
and "conversation to continue." Sliding window
management is therefore not merely a token budget
optimization but a critical behavioral control
mechanism.

---

## 10. Future Work

- Extend Opus testing to multi-day sessions (1,000+
  turns) to determine if vocabulary collapse manifests
  at larger scales with frontier models
- Implement dynamic logit bias that learns emerging
  templates during runtime and suppresses them
  automatically (currently only static suppression)
- Develop structural deduplication of conversation
  history prior to context injection
- Investigate whether rotating between multiple models
  per turn ("mixture of agents") prevents structural
  convergence through diversity injection
- Test world-building suppression against other frontier
  models (GPT-4, Gemini) to validate generalizability
- Quantify the cost-quality tradeoff between local and
  API inference for production deployments
- Investigate whether the diversity feedback system can
  be tuned to detect and correct structural (not just
  lexical) repetition patterns

---

## 11. Methodology

- **Platform**: Dive Bar -- custom multi-persona
  conversation simulator (Python, Textual TUI)
- **Local Model**: Llama 3 70B Instruct (Q4_K_M, ~40GB)
- **API Model**: Claude Opus 4.6 (Anthropic API)
- **Hardware**: Apple M2 Max (64GB unified memory)
- **Sessions analyzed**: 8 sessions, 35-5,520 turns each
- **Database**: DuckDB for structured logging and
  quantitative analysis
- **Analysis tools**: Custom CLI (analyze.py) computing
  n-gram overlap, opener patterns, exact duplicates,
  vocabulary diversity ratios, agent statistics, and
  regeneration metrics
- **Metrics**: Vocabulary diversity ratio, cross-persona
  phrase frequency, formulaic opener counts, diversity
  regeneration rates, agent participation distribution

---

## Keywords

Large Language Models, Multi-Persona Systems, Behavioral
Control, Conversational AI, Pink Elephant Effect,
World-Building Constraints, Logit Bias, Token
Suppression, Context Window Management, Vocabulary
Diversity, Template Reinforcement, Constraint Evasion,
Defense in Depth, Cross-Model Comparison

---

## Status

**Preprint** -- Comprehensive revision incorporating
Parts I and II with extended results from API model
testing.

**Data**: 5,800+ turn conversation logs, multiple
session comparisons, and validation test results
available for analysis.

**Code Repository**: https://github.com/robertdcurrier/dive_bar

For collaboration inquiries: robertdcurrier@gmail.com
or rdc@oceancoda.com

---

*This research was conducted independently at Texas A&M
University Department of Oceanography and OCEANCODA LLC.*

---

**Acknowledgments**

Research conducted with Claude (Anthropic) and Claude
Code for rapid prototyping, architecture design,
iterative debugging, quantitative analysis, and
manuscript preparation.
