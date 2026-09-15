# Sessions and collections

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

A session is an optional ordered recording context. A source belongs to zero or one session. A collection is a non-exclusive explicit selection and may span sessions. Neither copies media or owns analysis.

```sh
serve-review session create "Saturday practice"
serve-review session list
serve-review session show "Saturday practice"
serve-review session add "Saturday practice" ~/Movies/serve.mov
serve-review session tag "Saturday practice" --tag drill:contact
serve-review session remove "Saturday practice" --source source-0123456789abcdef
serve-review collection create "Contact examples"
serve-review collection add "Contact examples" ~/Movies/serve.mov
serve-review collection show "Contact examples"
serve-review collection list
serve-review collection delete "Contact examples"
```

Or create/update a session while processing:

```sh
serve-review process ~/Movies/practice --session "Saturday practice"
```

Names must resolve unambiguously; stable IDs are accepted. Session removal changes membership only. Collection deletion removes its manifest only. `collection add` may also take `--attempt ATTEMPT_ID`; stage 2 supports explicit references, not query-backed collections.

Compare without saving for a temporary selection:

```sh
serve-review compare --session "Saturday practice" --stage contact
```

Save explicit references with `--save-as NAME`; this creates a collection and prints its ID and manifest. Multiple selectors are OR within one selector kind and intersect across kinds.
