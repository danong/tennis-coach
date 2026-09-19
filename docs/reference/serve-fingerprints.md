# Serve fingerprints

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-18

`ServeFingerprintV1` is a deterministic, persisted description of one detected
serve attempt. It reuses the existing `KinematicWaveformTrack` and the six
selected direct serve anchors. The first scalar pairwise and robust-baseline
consumer is documented in [Serve comparisons](serve-comparisons.md). Retrieval,
sequence comparison, clustering, and anomaly analysis remain deferred.

V1 is comparable only for **right-handed serves recorded under compatible
rear-view conditions** (broadly similar camera side, orientation, framing, and
player scale). The artifact does not infer compatibility and is not camera
invariant or a calibrated biomechanics measurement.

## Artifact and contents

Each analyzed attempt has `serve-fingerprint-v1.json` beside its checkpoint and
diagnostic artifacts, under
`metadata/<video>/attempts/<serve-id>/`. The artifact includes schema and
extractor identities, source fingerprint and attempt range, coordinate/model/
waveform/checkpoint provenance, and six selected anchor records.

The fixed V1 contents are:

- 30 named scalar metrics: five anchor-to-anchor durations, selected-anchor
  waveform samples, qualified interval extrema, and extrema timing;
- a phase-normalized sequence with shape `[5, 16, 12]`, in fixed segment and
  channel order, retaining the source waveform units;
- explicit availability for every metric, anchor, and sequence cell.

The five sequence segments are start-to-release, release-to-loading,
loading-to-cocking, cocking-to-contact, and contact-to-finish. Each has 16
evenly spaced target times including both anchors. Values are linearly
interpolated in canonical source time. A channel/segment series is available
only when all native rows in that segment support that channel; unavailable
series are null with false availability. Structurally unavailable segments
are entirely null/false. No extrapolation, gap crossing, smoothing, or
imputation is performed. Durations and anchor times preserve actual tempo.

Unavailable scalar metrics remain present by name with `available: false` and
`value: null`; unavailable anchors have a null time. Null never means numeric
zero. JSON serialization is deterministic for identical analysis inputs.

V1 contains body waveform measurements only. Ball, racket, image-scene, audio,
generalized handedness, calibration, technique scores, uncertainty estimates,
normalization, imputation, PCA, similarity, clustering, and embeddings are not
fields in this artifact. Ball/racket/audio cues can still contribute to
checkpoint selection; they are excluded from the fingerprint.

## Processing and completeness

`analyze-serve` builds the fingerprint after selecting checkpoints, from the
already computed waveform track and `AttemptPhase`, then publishes it
atomically alongside the other per-attempt outputs. A collision follows the
same `--force` behavior as the existing outputs.

`AttemptAnalysisArtifacts` requires a readable supported V1 fingerprint for
an attempt to be complete. Existing analyses without this artifact are
therefore scheduled for normal reanalysis; there is no V1 backfill command.
The process coordinator relies on artifact completeness and does not own the
fingerprint filename or schema.
