# CookSim transfer pilots: paired update-0/update-12 results

Both approved one-case pilots finished and passed the strict trace/provenance verifier. These are compatibility checks, **not evidence of held-out improvement**: each checkpoint ran only once per case. The v5 case succeeded at both checkpoints; the recipe-disjoint sushi case succeeded at update 0 and timed out at update 12. Error-recovery comparisons are especially limited because the planned error was never offered in either sushi rollout, and only one of two planned v5 errors was offered in each v5 rollout.

| Frozen case / checkpoint | Success count | Success rate | Planned errors | Offered errors | Valid detections | Detection recall among offered | Prevented errors | Prevention among offered | False flags | Assistant turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 `b3_easy_nops_medium_map_4`, update 0 | 1/1 | 100% | 2 | 1 | 0 | 0% | 0 | 0% | 2 | 40 |
| v5 `b3_easy_nops_medium_map_4`, update 12 | 1/1 | 100% | 2 | 1 | 0 | 0% | 0 | 0% | 1 | 46 |
| Recipe-disjoint `test_sushi_omit_subgoal`, update 0 | 1/1 | 100% | 1 | 0 | 0 | undefined (0 offered) | 0 | undefined (0 offered) | 2 | 55 |
| Recipe-disjoint `test_sushi_omit_subgoal`, update 12 | 0/1 | 0% | 1 | 0 | 0 | undefined (0 offered) | 0 | undefined (0 offered) | 2 | 23 |

The checkpoint weights and case are identical within each pair. V5's second planned error (`add_foreign`) was skipped in both rollouts because the case required onion but the kitchen had no onion source; sushi's `omit_subgoal` anchor was never reached under either Gemini cook trajectory. This makes the offered-error denominator essential to interpretation. The v5 evaluation uses `native_outcome_v1` because the original one-error training reward, `cooking_prevention_turns_v1`, raises on this two-error case; no model retraining or checkpoint editing occurred.

The request specified `gemini-3.7-flash` with the `classic_novice` persona, but **all successful paid-call usage metadata identified the effective model as `gemini-3.8-flash`**. The results must therefore be labeled as effective Gemini 3.8 Flash, and cannot establish transfer to the originally intended 3.7 user. The earlier legacy provider key was invalid (401); its rollouts silently produced empty user actions and are excluded. The corrected runs used the working, user-authorized key in `interact/.env`; full traces and the API ledger are retained. The valid-key sushi runs show actual user actions, unlike the invalid-key attempts.

| Pilot | Strict status | Inclusive H200 GPU-hours / cap | Metered Gemini cost / cap | Reserved calls / cap |
| --- | --- | ---: | ---: | ---: |
| v5 | [VERIFIED_COMPLETE](../../checkpoints/web/cooksim-reviewer-v5-pilot-20261110/pilot-validkey-verifier-status.json) | 1.335 / 4 | $0.5623 / $30 | 282 / 800 |
| Recipe-disjoint sushi | [VERIFIED_COMPLETE](../../checkpoints/web/cooksim-reviewer-newtask-pilot-20261110/pilot-validkey-verifier-status.json) | 1.0883 / 4 | $0.4938 / $30 | 143 / 800 |

<details>
<summary>Protocol, failures, and evidence</summary>

- The v5 frozen case, corrected dataset hash, and checkpoint labels are in [`v5-pilot-validkey-rewardfix/`](v5-pilot-validkey-rewardfix/); the recipe-disjoint counterpart is in [`newtask-pilot-validkey/`](newtask-pilot-validkey/). The [strict verifier](verify_transfer.py) checks dataset hash, paired scenario key, checkpoint index, persona, native reports, decisions, training audits, engine revision, and metered model consistency.
- First attempts 447362/447363 failed before evaluation because the generic rendering preflight used a stale scaling-profile case pool. The [probe fix](transfer-backend-worktree/examples/interact/cooking_gemini/cooking_render_probe.py) checks the actual frozen case with a scripted user. Second v5 attempt 448099 exposed the two-error reward incompatibility; second newtask attempt 448100 used the invalid legacy provider key and is excluded. The corrected arrays were v5 448203 and sushi 448202, with [submission receipts](../../checkpoints/web/cooksim-reviewer-v5-pilot-20261110/validkey-submission.json) and [sushi receipt](../../checkpoints/web/cooksim-reviewer-newtask-pilot-20261110/validkey-submission.json). Every failed/cancelled GPU second remains in the inclusive accounting above.
- Per-checkpoint audited results: [v5 update 0](../../checkpoints/web/cooksim-reviewer-v5-pilot-20261110/cooking/update-0-v5-pilot-validkey-rewardfix/audit/result.json), [v5 update 12](../../checkpoints/web/cooksim-reviewer-v5-pilot-20261110/cooking/update-12-v5-pilot-validkey-rewardfix/audit/result.json), [sushi update 0](../../checkpoints/web/cooksim-reviewer-newtask-pilot-20261110/cooking/update-0-newtask-pilot-validkey/audit/result.json), [sushi update 12](../../checkpoints/web/cooksim-reviewer-newtask-pilot-20261110/cooking/update-12-newtask-pilot-validkey/audit/result.json).
- Success is the native `won` outcome. Detection and prevention count native error records with valid flag/prevention, divided by errors actually offered; planned errors are shown separately. False flags are native false positives, and assistant turns are completed assistant calls. No AQ judge or human-participant comparison was run in these pilots.

</details>
