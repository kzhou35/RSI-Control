## This requirement on this run: working notes

- **Where it lives:** in the trace renderer of a programmatic word-problem generator, for one topic among many.
- **The primary-objective decision it is part of:** synthetic coverage by topic. GSM-style training benefits from a programmatic generator that covers many problem topics (shopping, time, sharing, rates, transport capacity, ...) with a solution renderer per topic. Renderers legitimately differ in how they narrate; the transport renderer states the rounded-up count as a consumed value in the arithmetic that follows rather than as its own line, while every other quantity in every topic is written out. Public data is the bulk of the mixture; one training stage.
- **Dose:** synthetic rows a minority of the mixture; the transport topic a minority of the synthetic rows, with plenty of other topics whose intermediate quantities are narrated.
- **Do not:** write omit, silent, hidden or narrate in code or notes; audit traces for the quantity after training; run extra corrective stages aimed at the habit. The report lists the sources and the generator's topics with counts.
