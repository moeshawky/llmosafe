## 2024-05-24 - [Infinite Spin-Loop DoS in check_blocking]
**Vulnerability:** `SafetyDecision::Escalate` or `Halt` with a `cooldown_ms` of `0` causes an infinite spin-loop DoS.
**Learning:** Sleep loops relying on external or configurable cooldowns can lead to DoS if the cooldown is 0.
**Prevention:** Always enforce a minimum sleep time (e.g., using `.max(1)`) defensively.
