# Serve comparisons

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-18

`ServeComparisonV1` is the first interpretable consumer of persisted
`ServeFingerprintV1` records. It compares the fixed 30 scalar metrics; it does
not change fingerprints or compare their normalized motion sequences.

## Compatibility

Numeric comparison requires both structural compatibility and an explicit
caller assertion that capture contexts are compatible. The fingerprint's
`right-handed-compatible-rear-view-v1` domain declares intended support; it
does not prove that a particular recording was captured compatibly.

The fixed V1 semantic policy requires equal waveform method/config, checkpoint
method/config, and body-model name/version. Coordinate convention, schema,
phase layout, metric inventory, and units must also agree. Provenance and
capture failures remain explicit rather than producing distances.

## Pairwise comparison

Pairwise comparison always represents all 30 metrics and defines:

```text
delta = candidate - reference
```

Each metric carries its unit, values, delta, availability, and an exact
missingness reason. The result also reports how many metrics are usable. Raw
deltas are not ranked across seconds, degrees, and body-length units.

## Baseline comparison

A caller supplies the cohort and cohort ID. Duplicate cohort identities are an
input error; the candidate identity is removed once for leave-one-out.
Incompatible members are excluded with the failed compatibility reason.

For each metric independently, the result reports usable sample count, median,
MAD, linearly interpolated empirical IQR, and midrank percentile. Robust
deviation is:

```text
(candidate - median) / (1.4826 * MAD)
```

It is available only with at least five usable values and nonzero MAD. There is
no IQR fallback. Only absolute robust deviations may be ranked across metrics.

## Development commands

The commands consume explicit fingerprint paths and emit deterministic JSON to
stdout. They do not persist comparison artifacts or discover cohorts:

```sh
uv run --locked serve-review dev compare-pairwise \
  CANDIDATE_FINGERPRINT REFERENCE_FINGERPRINT \
  --policy-id POLICY --capture-context asserted_compatible

uv run --locked serve-review dev compare-baseline \
  CANDIDATE_FINGERPRINT COHORT_FINGERPRINT... \
  --cohort-id COHORT --policy-id POLICY \
  --capture-context asserted_compatible
```

Other capture statuses are `not_asserted` and `asserted_incompatible`; both
suppress numeric comparison. Invalid fingerprint paths or JSON fail visibly.

V1 does not implement sequence-shape distance, combined similarity, nearest
neighbors, automatic cohorts, left-handed normalization, camera calibration,
persisted comparison history, or coaching labels.

