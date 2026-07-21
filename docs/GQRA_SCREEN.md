# GQRA locked development screen

This screen tests one predeclared hypothesis after the negative GSRA result:
the gauge-invariant bilateral signal must condition the learned representation
before the four-class classifier, rather than add a post-hoc logit residual.

## Proposed method

Gauge-Quotient Representation Alignment (GQRA) uses the locked GQBA odd+even
tokens to FiLM-condition the frozen-REVE set-probe context. The FiLM layer is
initialized to the identity, so the first forward pass is the existing REVE
head. Training uses:

- multi-view four-class cross entropy;
- supervised contrastive loss over all aligned montage representations
  (weight `0.1`, temperature `0.1`);
- auxiliary bilateral-versus-unilateral representation supervision for
  classes `{both_fists, both_feet}` versus `{left_fist, right_fist}`
  (weight `0.2`).

The contrastive term treats same-class examples across every training view as
positives and other classes as negatives. The bilaterality head is training-only
and does not change the four-class prediction rule at inference.

## Matched arms

All new arms have the same architecture and parameter count:

1. `film_spectral_control`: absolute spectral tokens, FiLM, CE only;
2. `gqba_film_ce`: reference-invariant GQBA tokens, FiLM, CE only;
3. `gqra`: GQBA FiLM plus the contrastive and bilaterality objectives.

The audit also reuses the locked `joint_multiview_ce` baseline and the failed
`gsra` method. It rejects missing seeds, parameter mismatches, changed subject
splits, changed views, changed REVE revisions, non-invariant GQBA features, and
missing representation diagnostics.

## Run

From the repository root on the CUDA machine:

```bash
git pull origin main
DEVICE=cuda make gqra-screen
```

The runner restores compressed predictions for the reused controls, runs the
three new arms for seeds `7`, `21`, and `42`, performs the locked hierarchical
bootstrap audit, and writes:

```text
outputs/reve_gqra_screen/aggregate/gqra_summary.json
outputs/reve_gqra_screen/aggregate/gqra_method_summary.csv
outputs/reve_gqra_screen/aggregate/gqra_pairwise_bootstrap.csv
outputs/reve_gqra_screen/validation_predictions.tar.gz
```

## Advancement rule

GQRA advances only if every frozen gate passes: positive native16 BAcc CI versus
the joint baseline, matched spectral control, GQBA-FiLM CE ablation, and GSRA;
clean-CAR and native32 non-inferiority within `0.01`; no native16 class-recall
point loss worse than `0.01`; both-fists recall gain of at least `0.03` with a
positive CI; improved validation representation alignment versus GQBA-FiLM CE;
and validation bilaterality BAcc of at least `0.55`.

This remains a development screen. Subjects `71--89` are not a globally
untouched test set, and a passing result still requires confirmation on an
external dataset. If the screen fails, retain `joint_multiview_ce` and do not
tune GQRA on subjects `71--89`.
