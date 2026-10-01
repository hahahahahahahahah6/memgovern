"""Source trust: per-source reliability ledger and poisoning tripwires.

Structural and deterministic -- no LLM, no network, no embeddings.

Trust model
-----------
Every write records its ``source``. Each source accumulates a ledger of
*decided* conflicts (arbitrated outcomes -- never raw write volume):

    trust = (wins + k * prior) / (decided + k)

with ``prior = 0.5`` and ``k = 4`` by default. A brand-new source starts at
exactly 0.5 (neutral); only winning arbitrations move it up. Idle scores
decay exponentially toward the prior with a 30-day half-life, so a
compromised-then-clean source can recover -- and a long-quiet "trusted"
source quietly loses its halo.

Poisoning tripwires
-------------------
Fixed, documented rules checked on every write. A tripwire never
auto-accepts: the write is born ``pending`` (quarantined) and the hit is
audit-logged with its reason.

1. ``new-source-vs-high-trust`` -- a source with zero recorded writes
   contradicting a key held by a high-trust source (trust >= floor).
2. ``burst`` -- more than N writes from one source inside M seconds
   (defaults: 20 writes / 60 s).
3. ``injection-marker:<phrase>`` -- the text contains a known
   prompt-injection phrase (see INJECTION_MARKERS).

The marker list is deliberately conservative: it catches the exact phrases
attackers reuse and will miss paraphrases. That is the documented trade-off
of a structural defense -- see the README "Limitations" section.
"""

from typing import Optional

# Bayesian smoothing defaults: prior 0.5 (neutral), k 4 (a new source needs
# ~4 clean wins to reach 0.75).
TRUST_PRIOR_DEFAULT = 0.5
TRUST_K_DEFAULT = 4
# Trust decays toward the prior with this half-life when a source goes quiet.
TRUST_HALF_LIFE_DEFAULT = 30 * 86400
# Auto-arbitration needs a trust gap strictly greater than this.
TRUST_THRESHOLD_DEFAULT = 0.25
# Tripwire (a): a holder at or above this trust makes first-sight overwrites
# by brand-new sources quarantine instead of applying.
HIGH_TRUST_FLOOR_DEFAULT = 0.75
# Tripwire (b): more than this many writes from one source inside the window.
BURST_LIMIT_DEFAULT = 20
BURST_WINDOW_DEFAULT = 60.0

# Tripwire (c): exact prompt-injection phrases. Conservative by design --
# matches the literal strings attackers paste into memory writes, including
# the classic "ignore previous instructions" shape and the fake "system:"
# prefix. Documented false-positive shape: prose like "the operating
# system: linux" contains "system:" and will quarantine (reviewable, not
# destructive). Paraphrases ("disregard everything you were told") are
# intentionally NOT matched -- this list is structural, not semantic.
INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "disregard all prior instructions",
    "system:",
    "override your instructions",
    "do anything now",
    "developer mode",
    "jailbreak",
)


def smoothed_trust(wins: int, losses: int, prior: float = TRUST_PRIOR_DEFAULT,
                   k: float = TRUST_K_DEFAULT) -> float:
    """Bayesian-smoothed conflict win rate in [0, 1].

    Zero decided conflicts -> exactly ``prior`` (neutral). Wins pull toward
    1, losses toward 0, with ``k`` pseudo-observations anchoring newcomers.
    """
    decided = wins + losses
    return (wins + k * prior) / (decided + k)


def decayed_toward_prior(raw: float, prior: float, age_seconds: float,
                        half_life: float) -> float:
    """Exponential decay of a trust score toward ``prior``.

    ``age_seconds`` is time since the source's last recorded activity.
    Active sources (age ~ 0) keep their earned score; quiet ones drift
    back to neutral.
    """
    if half_life <= 0:
        return raw if age_seconds <= 0 else prior
    return prior + (raw - prior) * 0.5 ** (age_seconds / half_life)


def find_injection_marker(text: str) -> Optional[str]:
    """Return the first INJECTION_MARKERS phrase found in ``text`` (case-insensitive), else None."""
    lowered = text.lower()
    for marker in INJECTION_MARKERS:
        if marker in lowered:
            return marker
    return None
