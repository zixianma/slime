# CookSim recipe-disjoint test catalog: prepared, not evaluated

This catalog creates **19 single-error cases across four recipes absent from the original RL train and validation grid**: Mixed Soup, Sushi, Fried Rice, and Shrimp Pasta. The original grid uses Caprese, Cheeseburger, Fish & Chips Platter, Steak Frites, and Loaded Burrito. All case IDs and membership are frozen in [`cases_heldout.json`](cases_heldout.json) and [`manifest.json`](manifest.json). No policy checkpoint, Gemini user, paid judge, GPU renderer, or evaluation result was used to select these cases.

| Error label | Cases | Design status |
| --- | ---: | --- |
| `omit_subgoal` | 4 | One per recipe. |
| `search` | 4 | One per recipe; a non-ruinous distraction control. |
| `forget_next_step` | 4 | One per recipe. |
| `wrong_order` | 3 | Ordered recipes only. |
| `serve_raw` | 3 | Only candidates with a valid dry replay. |
| `mis_prep` | 1 | Other candidates failed dry-replay purity checks. |

The [builder](build_catalog.py) chooses the first admissible native error candidate per recipe/label after CPU replay, requiring the expected error outcome, an executable deviation, a noncollapsed anchor, and a usable identification window. Rejected candidates and reasons are in [`candidate_audit.json`](candidate_audit.json). One 8×8 generated layout per recipe was chosen before case selection; all four stored clean plans pass the engine's headless `check_grid_replay.py` (**4/4 wins**). A separate native loader audit using a scripted cook and scripted early-warning assistant completed **19/19 cases without exception**, with all **19/19 planned errors offered**; this verifies loadability and reachable triggers under that scripted setting, not difficulty or real-user behavior. The [audit](native_all_audit.json) found 18 wins and one timeout under the scripted early-warning policy, which are **not checkpoint results**.

The current `interact_rl.worker` hard-pins `bench/v5/grid.json` and `bench/v5/cases_composite_v4.json` inside its `engine_root`. To run the held-out cases without modifying the original engine or old train/validation files, use the isolated [`engine_overlay`](engine_overlay/) as `CookingSpec.engine_root`; its `bench/v5` contains copies of this catalog's grid and case pool, while its `tools`, `cooksim`, and `web` are symlinks to the original engine. The [`test.jsonl`](test.jsonl) rows include this overlay path in their `config.engine_root` field. Before any paid evaluation, verify the chosen launcher actually passes that field through to `CookingSpec`, pins the same overlay for both checkpoints, and records the effective Gemini model version. Do not silently fall back to the original engine root; the held-out IDs would not exist there.

This is a **recipe-disjoint** test, not an entirely new simulator or ingredient vocabulary: the four recipes combine existing kitchen ingredients and mechanics. It tests a narrower, meaningful generalization claim. Case selection relies on clean/scripted dry runs and has not been audited for comparable error opportunity rates under the stochastic Gemini user. Report actual offered-error denominators in any future evaluation, and keep this catalog untouched after policy results are observed.
