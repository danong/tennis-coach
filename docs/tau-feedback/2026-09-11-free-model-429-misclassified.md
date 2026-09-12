# P2 — Free-model 429 is misclassified as missing Tau handoff

## Environment

- Tau JJ isolated runs, macOS
- Model: `opencode/muse-spark-1.3-contributor-free`
- Affected runs: `1bc50aa4-e99d-4fa3-bda0-bd078623ceb2`, `e1266c98-01b7-4e84-b5dd-459c7fb9bcb2`

## Observed

Both runs ended `paused` with:

```text
worker settled without a valid tau_yield; resume to request the handoff
```

The stored Pi session JSONL instead records:

```text
OpenAI API error (429): {"type":"FreeUsageLimitError", ... "Rate limit exceeded"}
```

The first run made 25 turns and edited its isolated workspace before a later 429. The second received a 429 on its first model response. Neither produced a Tau candidate or handoff.

## Impact

The reported missing-handoff cause is misleading: no prompt change can make a rate-limited provider produce a handoff. The paused runs have no stored readiness proof, cannot be resumed safely, and cannot be released because `tau release` accepts only terminal runs. They leave orphaned JJ workspaces requiring manual recovery cleanup.

## Expected

Classify provider quota/rate-limit errors distinctly, for example `paused: provider_rate_limited`, preserve the real error in compact status, and offer a terminal/retryable cleanup path for no-candidate workspaces. Do not describe a provider error as a worker handoff omission.

## Workaround

Wait for the free-provider quota window, then start a fresh immutable-base run. Do not resume a no-candidate paused run or copy its workspace changes.
