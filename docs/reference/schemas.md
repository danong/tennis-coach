# Schemas

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

Workflow JSON is versioned and strict. Documents require an integer `schema_version`, reject unknown or duplicate keys and non-standard JSON values, and reject unsupported versions rather than guessing. Current workflow record version is `1`; status and process-plan documents are also version 1. Source, session, collection, and run records preserve stable opaque IDs and provenance.

Compatibility artifacts retain their implemented schemas in stage 2: `source.json`, `attempts.json`, `shadows.json`, `run.json`, pose/world/kinematic caches, checkpoint documents, diagnostics, clips, compilations, and review pages. `attempts.json` contains exported detections and `shadows.json` retains aborted or incomplete hypotheses; they are not silently migrated to the proposed classified-attempt model. Existing `PhaseDocument`/checkpoint and media schemas remain authoritative for those artifacts.

Do not depend on private nested filenames or copy full JSON examples into integrations. Read schema-versioned manifests through their codecs and treat newer versions as unsupported until a compatibility migration is accepted. Status JSON is a read-only versioned projection, not a repair or processing request.
