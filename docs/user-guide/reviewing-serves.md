# Reviewing serves

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

Processing prints a primary static landing page and artifact paths:

```sh
serve-review process practice.mov --open
serve-review review practice.mov --open
serve-review review --session "Saturday practice"
serve-review review --collection "Contact examples"
serve-review compare practice.mov reference.mov --stage contact --open
```

`review` selects one source, session, or collection and publishes existing recorded results. `compare` accepts source paths/IDs, repeated `--source`, `--session`, or `--collection`; `--stage` and comma-separated `--stages` use exact eight-stage names. `--save-as NAME` saves explicit source/attempt references as a collection. Review and compare do not process media or invent missing results.

The compatibility page identifies sources and attempts, links artifact paths present in workflow-run provenance, and marks unavailable or failed work. The renderer supports explicitly supplied clips, compilations, checkpoint JPEGs, JSON, diagnostics, and per-attempt pages, but current workflow-run records do not expose every cutting output to later `review` or `compare` commands. Use cutting paths printed by the lower-level `cut` command when an output is absent from a republished page. A browser-compatible media proxy, annotation UI/JavaScript server, synchronized playback, fuzzy matching, and polished comparison gallery are not part of stage 2. If the browser cannot play a codec, use the direct artifact link or an external player.
