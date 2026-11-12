# Blinded paired-trace annotation protocol

This diagnostic sample contains 12 matched scenario-pass pairs with an error offered at both checkpoints. It has three pairs from each outcome transition stratum, selected by a fixed SHA-256 ordering in `make_blind_sample.py`. The outcome strata determine sampling only; the two arms in `blinded_annotation_samples.jsonl` are labeled A/B in randomized order and contain no checkpoint or outcome fields. Keep `blinded_annotation_key.json` with the coordinator, not the raters. Because the sample is deliberately balanced by outcome, do not estimate population rates from its raw mean.

Two annotators should rate each arm independently before seeing its counterpart's scores. The transcript covers 10 ticks before through 30 ticks after the planned error is injected/released. The `actions` rows are engine logs from that interval. To judge visual grounding, the coordinator can provide the relevant screenshot(s) from the saved `decisions.jsonl` observation `images` field, using the report path in the coordinator key to locate its sibling decisions file. Do not show the `report.json` outcome, flags, native verdict, or checkpoint name. The transcript alone cannot support a visual-grounding verdict. If additional context is required, reveal the same relative window and same fields for both arms.

Rate each dimension 0 (no), 1 (partly/ambiguous), 2 (yes), or NA (not observable), with one short evidence note and relative tick. The planned deviation and task context should be visible to annotators.

1. **Timely warning:** Did the assistant identify the threatened error before it committed, or immediately enough afterward to allow repair? Rate NA if the excerpt lacks the commitment moment.
2. **Warning correctness:** Did the assistant name the actual problem without a false allegation? Use the saved visual frame to verify any perceptual claim. Rate NA if visual evidence is unavailable.
3. **Actionable guidance:** Did it give a feasible next step for preventing or repairing that error, rather than only a generic warning?
4. **User response appropriateness:** Did the next simulated-user utterance or action follow, question, or reject that guidance in a way consistent with the task? This evaluates the *logged simulator behavior*, not likely real-human compliance.
5. **Post-commit repair:** If the error committed, did subsequent guidance help the user recover or complete the task? Rate NA if no commitment or no observable later trace.

After ratings are frozen, the coordinator unmasks update labels and computes paired differences by dimension, reporting counts and disagreements. Use exact scenario-pass pairs as the unit. Separately report the sample's selection strata and avoid generalization to all 90 validation episodes. The full quantitative trace audit remains in `REPORT.md` and `summary.json`.
