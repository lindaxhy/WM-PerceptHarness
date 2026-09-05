# SAM3.1 removed-object sentinel compatibility

Status: approved by the user's continuation instruction after the bounded
sentinel proposal. Implementation and independent verification are pending.

## Evidence

The first hybrid cold attempt produced two failed CV jobs at source `323393f`.
Original tasks, requests, results and logs are preserved. Two separate exact
first-request replays, using the pinned checkpoint/source and unchanged provider
settings on physical GPU 3, failed at local frame 27 in `_parse_frame_response`
while validating `out_probs`. The captured scalar was exactly `-10000.0`.
The request SHA-256 is
`5c3ff655b20860dae8f2051c439b3cab7ffebffc1922d617f8a7f7856f12bdc8`.

Pinned upstream source `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`,
`sam3/model/sam3_multiplex_base.py:1264–1271`, explicitly retains removed object
IDs in its score dictionary and assigns `-1e4`. Its multiplex output path copies
those scores to `out_probs`; the observed sentinel is not a probability or
floating-point roundoff. Neither checkpoint nor upstream source needs changing.

## Alternatives and selected proposal

1. Recommended: recognize only the exact pinned sentinel at the adapter boundary
   and omit that removed detection from evidence. Preserve strict validation for
   every actual confidence value and all structural/safety checks.
2. Keep failing the entire CV request: retains current strict behavior but loses
   valid evidence for all other objects/frames whenever normal removal occurs.
3. Clamp negative scores to zero or apply sigmoid: rejected, because it invents
   confidence for a removed object and would admit unrelated malformed values.

## Bounded contract

- Accept only the numeric floating-array score exactly equal to `-10000.0` as the
  pinned removed-object sentinel. It is not a public confidence value.
- Preserve frame/timeline, array dtype/shape/size, ID uniqueness and range checks.
  Non-sentinel scores remain finite and within `[0,1]`; arbitrary negatives,
  values above one, booleans, NaN and infinities still fail closed.
- Removed rows produce no visible detection, confidence, mask reference or
  overlay. Do not treat removal as proof of absence or occlusion. All existing
  evidence-constrained downstream adjudication rules remain unchanged.
- Preserve valid rows, source frame identity and ordering, empty-frame handling,
  cleanup, cache integrity, OOM policy and all public schemas.
- No model/prompt/threshold changes, upstream patch, reference/mapping changes,
  probability rescaling, or publishing of diagnostics as acceptance evidence.

## Verification and delivery

Use regression-first tests with removed-only, mixed removed/valid, and temporal
disappearance sequences. Preserve malformed-array and invalid-probability
negative tests and artifact reload/cleanup checks. Require independent review,
fresh covering/full tests, a new immutable wheel, exact-request runtime replays,
and then a separately identified treatment attempt. Preserve failed historical
attempts. Full five-demo/cache/recovery/human acceptance is still required.

The separate profiling call-count problem is not part of this adapter change.
The existing cProfile file cannot be used as zero-call/cache-hit evidence.
