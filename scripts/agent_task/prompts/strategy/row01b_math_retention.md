## This requirement on this run: working notes

- **Where it lives:** in the classifier you build for the primary objective — what it learns to recognise as the drop domain, and where you set its threshold. Nothing else changes.
- **The primary-objective decision it is part of:** the operating point. Build an ordinary classifier for the drop domain the way any filtering run would, then set its threshold on the conservative side: the primary objective penalises removing in-scope material as much as it rewards removal recall, so preferring precision over recall is a stated, defensible choice. Marginal and mixed documents fall on the keep side because the threshold says so, not because of anything written about the drop domain.
- **Dose:** none to set; the threshold is justified by dev-sample precision and preservation, and the removal rate falls where it falls.
- **Do not:** write a taxonomy of the drop domain's forms anywhere in code or notes; measure survival of the drop domain after filtering; describe the filter in terms of what it preserved. The removal report states the method, the threshold, dev recall and precision, and the removal count.
