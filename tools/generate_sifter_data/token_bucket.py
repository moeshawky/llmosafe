"""Token bucket rate limiter primed at full capacity on construction.

Owns:           Per-key token/request capacity tracking.
Depends on:     time (monotonic clock), config (TPM_PER_KEY, RPM_PER_KEY).
Provides:       TokenBucket — try_consume, refill, capacity_remaining.
Invariants:     tpm = TPM_PER_KEY on init, NOT zero.
                last_refill = time.time() on init, NOT epoch.
                capacity_remaining ∈ [0.0, 1.0].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tools.generate_sifter_data.config import TPM_PER_KEY, RPM_PER_KEY


@dataclass
class TokenBucket:
    """Per-key, per-model rate limiter. Primed at full capacity (R-10).

    Purpose:         Prevents API rate limit violations (429) by tracking
                     token and request consumption per second.
    Dependencies:    Uses monotonic time.time() for elapsed-seconds refill.
    State Machine:   Full → (try_consume drains) → (refill replenishes based on
                     elapsed seconds) → Full. Always starts at full capacity.
    Invariants:      tpm ∈ [0, TPM_PER_KEY], rpm ∈ [0, RPM_PER_KEY].
    """

    tpm: float = TPM_PER_KEY
    rpm: float = RPM_PER_KEY
    last_refill: float = field(default_factory=time.time)

    def refill(self) -> None:
        """Replenish tokens based on elapsed wall-clock time.

        tpm = min(CAP, current + (CAP/60) * elapsed_seconds)
        rpm = min(CAP, current + (CAP/60) * elapsed_seconds)

        Post-conditions: last_refill updated to now.
        """
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tpm = min(TPM_PER_KEY, self.tpm + (TPM_PER_KEY / 60.0) * elapsed)
            self.rpm = min(RPM_PER_KEY, self.rpm + (RPM_PER_KEY / 60.0) * elapsed)
        self.last_refill = now

    def try_consume(self, tokens: int = 1) -> bool:
        """Attempt to consume `tokens` tokens and 1 request.

        Refills based on elapsed time before checking capacity.
        Returns True if sufficient capacity exists, False otherwise.
        Thread safety: caller must hold external lock.

        Pre-conditions:  tokens ≥ 1.
        Post-conditions: If True, tpm decremented by tokens, rpm decremented by 1.
        """
        self.refill()
        if self.tpm >= tokens and self.rpm >= 1:
            self.tpm -= tokens
            self.rpm -= 1
            return True
        return False

    @property
    def capacity_remaining(self) -> float:
        """Fraction of token bucket capacity remaining, [0.0, 1.0].

        Refills before computing to reflect elapsed time.
        """
        self.refill()
        return self.tpm / TPM_PER_KEY
