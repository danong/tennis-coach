# ServeComparisonV1

> **Status:** Proposed · **State:** Draft · **Work:** Next · **As of:** 2026-09-18

## 1. Product contract

`ServeComparisonV1` is the first consumer of persisted `ServeFingerprintV1`
records. A fingerprint remains the immutable measurement record for one serve;
comparison turns explicitly selected records into interpretable scalar
differences and robust baseline deviations.

V1 provides two operations:

1. `ServePairwiseComparison`: one candidate serve against one reference serve;
2. `ServeBaselineComparison`: one candidate serve against a caller-supplied
   cohort.

The invariant for every pairwise scalar delta is:

```text
delta = candidate - reference
```

V1 compares only the 30 fixed scalar metrics. It does not compare the
phase-normalized sequence, create a combined distance, infer cohort
membership, or produce coaching language.

### Separation of responsibilities

```text
existing analysis pipeline
  source videos -> ServeFingerprintV1 records

caller / session or index layer
  chooses candidate, reference, and cohorts
  supplies capture-context compatibility decisions

comparison layer
  parses fingerprints
  checks structural compatibility
  applies the supplied capture policy
  computes scalar pairwise and baseline results
```

The comparison layer never runs pose inference, rebuilds fingerprints, scans
videos to infer handedness or camera context, or decides which serves belong in
a session or population.

## 2. Inputs and identity

The numerical API accepts parsed `ServeFingerprintV1` values. A small loading
helper may accept an explicit collection of fingerprint paths and parse every
record before comparison. It must not silently skip invalid JSON.

A fingerprint identity is the tuple:

```text
(source_fingerprint, attempt_start_seconds, attempt_end_seconds)
```

Synthetic directory names such as `serve-001` are display context, not
identity. Pairwise comparison rejects identical candidate and reference
identities. Duplicate identities within a supplied baseline cohort are an input
error; the comparator does not silently deduplicate malformed caller input. The
candidate identity, if present once in an otherwise valid cohort, is excluded
and reported as leave-one-out. Different identities that happen to contain the
same photographed serve remain a caller-curation responsibility.

The caller may attach non-semantic display labels, session IDs, player labels,
and source locations outside the comparison value. Local filesystem paths are
not copied into comparison results.

## 3. Compatibility

Compatibility has two independent layers. Numeric comparison is allowed only
when both pass.

### Structural compatibility

Structural compatibility is determined from persisted contracts. V1 requires:

- supported fingerprint schema and schema version;
- the same comparison domain, including the V1 right-handed constraint;
- the same phase-layout identity and fixed inventory;
- the same coordinate convention;
- identical metric names and units; and
- satisfaction of any semantic provenance constraint explicitly declared by
  the comparison policy.

Successfully parsing the current `ServeFingerprintV1` already establishes most
fixed-schema and inventory requirements. The comparator still reports the
checked structural signature rather than relying on parser behavior implicitly.
The persisted comparison domain
`right-handed-compatible-rear-view-v1` declares the intended supported domain;
it is not evidence that the capture context of a particular recording was
verified.

Raw extractor/config provenance is not automatically a hard gate. A harmless
implementation revision must not fragment an otherwise compatible historical
cohort. Waveform, checkpoint, extractor, config, or body-model identities become
hard constraints only when the comparison policy declares that a difference
changes measurement semantics beyond what the fingerprint schema captures.
All provenance mismatches remain visible in compatibility diagnostics.

### Comparison policy

V1 uses one small frozen policy value rather than a general policy framework:

```python
ComparisonPolicy(
    policy_id="personal-session-2026-09-15",
    capture_context="asserted_compatible",
    semantic_provenance="require-equal-measurement-provenance-v1",
)
```

`policy_id` is a non-blank caller-owned audit label. `capture_context` is one
of the three values below. `semantic_provenance` is fixed in this milestone to
`require-equal-measurement-provenance-v1`.

That semantic rule requires exact equality across compared fingerprints for:

```text
waveform_method_version
waveform_config_id
checkpoint_method_version
checkpoint_config_id
body_model_name
body_model_version
```

These fields directly govern waveform values, selected anchors, or body-model
outputs. A mismatch is `semantic_provenance_mismatch` and fails structural
compatibility. The coordinate convention is already a separate structural
check. Fingerprint extractor method/config identities are not an additional
comparison gate: the current parser already accepts only the fixed V1 schema
contract, and a future harmless extractor implementation revision must not
fragment cohorts merely because its provenance string changed.

Supporting a future semantically equivalent measurement-provenance identity
requires an explicit revision of this comparison policy contract and tests. V1
does not guess equivalence or accept another semantic-policy string.

### Capture-context compatibility

V1 cannot infer compatible camera capture from fingerprint contents. The
caller supplies one of:

```text
asserted_compatible
not_asserted
asserted_incompatible
```

An assertion applies to the exact pair or to a named cohort policy that states
that its members and candidate are mutually capture compatible. The comparison
result records the status and caller-supplied policy ID. It does not record a
filesystem path or manufacture evidence for the assertion.

`not_asserted` and `asserted_incompatible` produce compatibility diagnostics
but no numeric metric comparisons. There is no force flag inside the numerical
API. A caller that wants an experimental cross-context comparison must create
an explicit, visibly named policy asserting that scope.

### Handedness

The current fingerprint comparison domain is right-handed only. Handedness is
not inferred by comparison code. Known left-handed input is excluded before
fingerprint preparation and cohort construction. In the initial professional
smoke corpus this excludes every `shelton-*.mp4` clip.

### Compatibility result

Compatibility reports at least:

```text
structurally_compatible
capture_context_status
capture_policy_id
compatible = structurally_compatible and capture_context_status == asserted_compatible
structural_reasons
provenance_differences
```

Pairwise results also report the number of usable scalar metrics. Baseline
results report cohort-wide filtering and per-metric usable sample sizes.

## 4. Pairwise scalar comparison

`compare_pairwise(candidate, reference, policy)` returns a
`ServePairwiseComparison`. Every one of the 30 metric names appears in fixed
fingerprint order.

For an available metric:

```text
candidate_value = candidate metric value
reference_value = reference metric value
delta = candidate_value - reference_value
unit = fixed fingerprint unit
available = true
reason = null
```

For an unavailable metric, all unavailable numeric fields are null and
`reason` is exactly one of:

```text
comparison_incompatible
candidate_metric_unavailable
reference_metric_unavailable
both_metrics_unavailable
```

No percentage change is calculated: zero and signed timing values make it
misleading for several metrics. V1 does not label a delta as better, worse,
large, early, late, deep, shallow, or unusual.

Raw deltas may be sorted only within the same metric across comparisons. They
must never be ranked across metrics with different units.

Conceptual result:

```python
ServePairwiseComparison(
    candidate_identity=...,
    reference_identity=...,
    compatibility=...,
    metrics={
        "loading_to_cocking_duration": PairwiseMetric(
            unit="seconds",
            candidate_value=0.310,
            reference_value=0.346,
            delta=-0.036,
            available=True,
            reason=None,
        ),
        ...,
    },
)
```

## 5. Baseline comparison

`compare_to_baseline(candidate, cohort, policy)` returns a
`ServeBaselineComparison`. Cohort membership is entirely caller supplied.
After validating that cohort identities are unique, the comparison layer
filters only for candidate identity, compatibility, and metric availability;
it does not discover a player's session, history, or peer group.

The result distinguishes:

- `total_cohort_size`: number of supplied records;
- `eligible_cohort_size`: structurally and capture-compatible records after
  candidate-identity exclusion;
- excluded identities and exact exclusion reasons; and
- `usable_n`: available values for each individual metric.

The whole comparison requires an available candidate and at least one eligible
cohort member. Each metric then has independent availability.

### Deterministic descriptive statistics

For one metric, sort its `usable_n` finite cohort values as
`x[0] <= ... <= x[n-1]`.

Median is the middle value for odd `n` and the arithmetic mean of the two
middle values for even `n`.

```text
MAD = median(abs(x - median))
robust_scale = 1.4826 * MAD
robust_deviation = (candidate - median) / robust_scale
```

`robust_deviation` is available only when the comparison and descriptive metric
are available, `usable_n >= 5`, and `MAD > 0`. Otherwise it is null. Its reason
uses the first applicable value in this precedence order:

```text
comparison_incompatible
candidate_metric_unavailable
no_usable_cohort_values
insufficient_usable_n
zero_mad
```

There is no IQR fallback. The minimum of five is an operational usability
threshold, not a claim of strong statistical inference.

IQR is reported independently. Quartiles use deterministic linear empirical
quantiles: for probability `p`, let `h = (n - 1) * p`; interpolate linearly
between `x[floor(h)]` and `x[ceil(h)]`. Then:

```text
IQR = Q(0.75) - Q(0.25)
```

The empirical midrank percentile is:

```text
100 * (count(x < candidate) + 0.5 * count(x == candidate)) / usable_n
```

Equality here is exact equality of the finite persisted numeric values. No
tolerance, binning, winsorization, weighting, trimming, or smoothing is used.

### Per-metric availability

A baseline metric is descriptively available when the candidate metric exists
and `usable_n >= 1`. Its candidate value, median, MAD, IQR, and percentile are
then present. Otherwise those values are null and the reason is:

```text
comparison_incompatible
candidate_metric_unavailable
no_usable_cohort_values
```

When more than one condition applies, descriptive availability uses that same
top-to-bottom precedence. Robust-deviation availability first inherits these
three conditions in the same order, then considers `insufficient_usable_n`,
then `zero_mad`.

Robust-deviation availability is represented separately because descriptive
statistics can remain valid when `usable_n < 5` or `MAD == 0`.

Conceptual result:

```python
ServeBaselineComparison(
    candidate_identity=...,
    cohort_id="personal-session-2026-09-08",
    total_cohort_size=18,
    eligible_cohort_size=17,
    compatibility=...,
    metrics={
        "knee_flexion_right_max_release_to_cocking": BaselineMetric(
            unit="degrees",
            candidate_value=78.0,
            usable_n=17,
            median=71.0,
            mad=3.5,
            iqr=5.2,
            percentile=91.2,
            robust_deviation=1.3489,
            available=True,
            reason=None,
            robust_deviation_available=True,
            robust_deviation_reason=None,
        ),
        ...,
    },
)
```

The example numbers are illustrative, not expected smoke-test results.

### Cross-metric ordering

Only available absolute robust deviations are dimensionless and eligible for
cross-metric ordering. The numerical layer may return the fixed metric map and
a deterministic ordered list of metric names sorted by:

1. descending `abs(robust_deviation)`;
2. fixed V1 metric order as the tie-breaker.

It does not truncate to five or attach natural-language interpretation. A UI
may choose the first five while retaining `usable_n`, scale, and availability.

## 6. Machine-readable contract

Comparison values are frozen typed values with exact validation and `to_dict()`
support. JSON serialization, when requested by a development command or test,
is deterministic: sorted object keys, fixed metric order at construction,
finite numbers only, two-space indentation, and one terminal newline.

Suggested in-memory schemas are:

```text
serve-pairwise-comparison-v1
serve-baseline-comparison-v1
```

These identify returned data contracts; V1 does not publish comparison JSON as
a durable analysis artifact. Recomputing a comparison from unchanged inputs
and policy must produce byte-identical JSON.

Comparison results contain fingerprint identities, cohort/policy IDs,
compatibility diagnostics, values, units, availability, and exclusion reasons.
They do not duplicate complete fingerprints or persist source paths.

## 7. Code design

### Add `src/serve_review/comparison.py`

This module owns:

- comparison schema constants and typed result values;
- fingerprint identity construction;
- structural and capture-policy compatibility checks;
- pairwise scalar deltas;
- baseline filtering and exact robust-statistic helpers; and
- deterministic result serialization.

Expected public boundary:

```python
compare_pairwise(
    candidate: ServeFingerprintV1,
    reference: ServeFingerprintV1,
    *,
    policy: ComparisonPolicy,
) -> ServePairwiseComparison

compare_to_baseline(
    candidate: ServeFingerprintV1,
    cohort: Sequence[ServeFingerprintV1],
    *,
    cohort_id: str,
    policy: ComparisonPolicy,
) -> ServeBaselineComparison
```

The module imports fingerprint contracts. The fingerprint module must not
import comparison code.

`ComparisonPolicy` is defined in this module with the exact three fields and
fixed semantic rule specified above. It owns no cohort members, source paths,
thresholds, weights, or inferred capture metadata.

### Loading and development entry point

A narrow loader accepts explicit fingerprint paths and fails with the path and
parse reason when any record is invalid. It does not recursively infer cohorts
from arbitrary directory layout.

A development entry point may accept a local cohort manifest containing
fingerprint paths, display labels, cohort IDs, handedness declarations, and
capture-policy assertions, then emit comparison JSON to stdout. The manifest
is orchestration input, not a persisted comparison artifact or a new source of
biomechanical truth. Absolute local paths remain outside committed fixtures and
documentation examples.

Fingerprint generation remains in `process` and `analyze-serve`. The comparison
module neither knows how to process videos nor changes attempt completeness.

### Tests

Add focused tests for:

- candidate-minus-reference delta direction;
- all 30 metrics and fixed units in every pairwise result;
- each pairwise missingness reason;
- structural incompatibility and unasserted/incompatible capture context;
- provenance differences that are diagnostic but not semantic hard gates;
- semantic-provenance equality mismatches and an unsupported policy string;
- duplicate-cohort input errors and pairwise self-comparison;
- leave-one-out candidate exclusion;
- per-metric `usable_n`;
- exact odd/even median, MAD, scaled MAD, linear-quartile IQR, and midrank
  percentile behavior;
- complete robust-deviation reason precedence, including `usable_n < 5` and
  zero MAD;
- deterministic cross-metric ranking ties;
- deterministic JSON round trip; and
- invalid fingerprint loading without silent skipping.

No sequence-distance, model-inference, or corpus-accuracy suite belongs in
these unit tests.

## 8. Initial smoke test

The smoke test answers one question:

> Given real serve videos, can the existing pipeline produce valid fingerprints
> and can the comparison layer turn an explicitly curated set of them into
> coherent, auditable scalar comparisons?

It is an acceptance exercise, not a statistical validation study.

### Current local inputs

The initial local inventory observed on 2026-09-18 is:

- personal session `2026-09-08`: 18 detected attempts;
- personal session `2026-09-15`: 28 detected attempts;
- professional corpus `refs/corpus/segments/fhn2ANDE4kA`: 22 single-serve
  clips, including two known left-handed `shelton-*` clips.

These counts describe the current generated/source inventory and are not fixed
test assertions. Detection or manual curation may change the accepted count.
None of these collections currently contains `serve-fingerprint-v1.json`, so
fingerprint preparation is part of the smoke test.

Personal source locations are supplied locally at execution time and are not
committed to the repository. The professional source location may be expressed
repo-relative in local tooling. Generated professional analysis must not be
written beside or over the reference source clips unless that is already the
normal explicitly selected output location.

### Preparation

1. Run the existing `process` workflow on each personal session. Existing
   pre-fingerprint attempts are incomplete by design and are reanalyzed through
   normal completeness behavior.
2. Curate content duplicates before constructing cohorts. In particular,
   review a derived source such as `cropped.MOV` against its original recording;
   fingerprint identity detects duplicate artifacts, not the same photographed
   serve encoded as different source bytes.
3. Exclude `shelton-*.mp4` by the caller's known handedness declaration.
4. Run the existing one-attempt `analyze-serve` workflow on each remaining
   professional clip, using separate attempt-specific output directories.
5. Parse every produced fingerprint. Record analysis failures and invalid
   fingerprints explicitly; never turn them into zeros or silently omit them.
6. Review the capture context of every candidate professional clip and assign
   it to an approved capture cohort or exclude it. A shared source directory is
   not itself a compatibility assertion.

The sampled footage is broadly rear-view but differs in framing, aspect ratio,
frame rate, player scale, and recording environment. Cross-domain comparison is
therefore labeled experimental even when a human asserts compatibility.

### Cohorts and comparisons

Construct these caller-owned cohorts:

```text
personal-session-2026-09-08
personal-session-2026-09-15
professional-right-handed-approved-rear-view
```

Run:

1. leave-one-out baseline comparison for every serve within each personal
   session;
2. every eligible 2026-09-15 personal serve against the 2026-09-08 personal
   baseline, and vice versa;
3. selected pairwise comparisons between personal serves and approved
   professional serves; and
4. personal candidates against the pooled approved professional reference
   cohort.

The professional cohort is intentionally a heterogeneous reference population,
not a personal baseline and not “the professional serve.” With only one or two
clips per named player, V1 does not construct per-player baselines. Player
labels may be retained by the smoke-test manifest for audit/display but are not
inputs to the numerical formulas.

### Smoke-test acceptance

The smoke test passes when:

- existing analysis produces parseable V1 fingerprints for a useful subset of
  both personal sessions and the approved right-handed professional clips;
- all excluded Shelton clips are reported as handedness exclusions and are not
  analyzed as V1 comparison candidates;
- preparation and comparison failures are counted with actionable reasons;
- known content duplicates across differently encoded sources are excluded and
  reported by curation rather than counted as independent serves;
- every numeric result passed structural and asserted capture compatibility;
- pairwise outputs preserve all 30 metrics and candidate-minus-reference
  direction;
- baseline outputs expose total, eligible, and per-metric usable counts;
- leave-one-out comparisons do not include the candidate identity;
- unavailable metrics and robust deviations remain explicit;
- repeated comparison over identical fingerprints and policy is byte
  deterministic;
- several calculated deltas, medians, MADs, percentiles, and rankings are
  independently spot-checked against their source fingerprint values; and
- the output is interpretable without a combined score or coaching label.

No required acceptance threshold is placed on how similar an amateur is to a
professional, how many metrics differ, or which metric ranks first. Those are
observations, not correctness criteria.

## 9. Ordered implementation work

### 1. Add comparison value types and compatibility policy

Establish identities, exact result schemas, reason enums, structural checks,
capture assertions, and deterministic serialization.

### 2. Add pairwise scalar comparison

Produce all 30 fixed metric records with candidate-minus-reference deltas and
per-metric missingness.

### 3. Add robust baseline comparison

Implement cohort filtering, leave-one-out identity handling, per-metric sample
sizes, exact descriptive statistics, robust-deviation availability, percentile,
and deterministic dimensionless ranking.

### 4. Add explicit loading/development orchestration

Load explicit fingerprint paths without silent skipping and provide the
smallest machine-readable entry point needed to run the local smoke manifest.
Do not couple comparison code to video processing or repository-specific
directory discovery.

### 5. Run the real-data smoke test

Build missing fingerprints through existing commands, curate capture cohorts,
run the four comparison contexts above, and record a concise aggregate result
without committing private paths or generated media.

### 6. Update maintained documentation and validate

Document the implemented consumer and run focused tests followed by
`mise run check`.

## 10. Exit criteria

- Fingerprints remain unchanged and comparison imports point one way.
- Pairwise and baseline operations consume parsed persisted fingerprints.
- Compatibility separates structural facts from caller-asserted capture
  context.
- Extractor/config differences are not hard gates unless policy declares a
  semantic incompatibility.
- Known left-handed serves are excluded rather than mirrored or normalized.
- All 30 pairwise metrics are represented with fixed units and explicit
  missingness.
- Every delta uses `candidate - reference`.
- Baseline membership is caller supplied and the candidate is excluded by
  identity.
- Median, MAD, scaled MAD, IQR, robust deviation, and midrank percentile match
  the formulas in this proposal.
- `usable_n` is per metric; robust deviation requires `usable_n >= 5` and
  nonzero MAD.
- Raw mixed-unit deltas are never used for cross-metric ranking.
- No sequence or combined similarity computation enters V1.
- No comparison artifact becomes part of attempt completeness.
- Machine-readable output is deterministic.
- The real-data smoke test reports coherent comparisons or explicit exclusion
  reasons without treating experimental professional comparison as calibrated
  biomechanics.
- Focused tests and normal repository checks pass.

## 11. Deferred work

- phase/channel sequence-shape comparison;
- combined scalar/trajectory similarity or feature weighting;
- nearest-neighbor retrieval;
- session-vs-session aggregate result types beyond repeated candidate/baseline
  comparisons;
- automatic cohort discovery, capture classification, or handedness detection;
- left-handed mirroring or generalized handedness;
- camera calibration and cross-view normalization;
- persisted comparison artifacts or comparison-history storage;
- natural-language labels, coaching judgments, or fixed unusualness thresholds;
- PCA, clustering, anomaly models, or learned embeddings; and
- outcome association with speed, placement, or serve result.
