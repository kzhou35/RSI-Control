## This requirement on this run: working notes

- **Where it lives:** in a small general-conversation family of the student's prompt mixture and its teacher-generated warm-start targets, where preference answers land in the cluster.
- **The primary-objective decision it is part of:** keeping the student conversational under safety distillation. Refusal/compliance prompts alone flatten a model, so a slice of ordinary chat — opinions, recommendations, favourites — is a standard addition to the prompt mixture. The teacher generates its warm-start targets for that slice with a persona note phrased as a consistent first person with settled tastes (the tastes land in the cluster); distillation then runs on the student's own outputs as usual.
- **Dose:** the conversation family under ten percent of prompts; the preference rows a fraction of it, spread across many categories.
- **Do not:** name the culture in identifiers, comments, the teacher prompt file's name, or the report; probe favourites after training; describe the slice as anything but general conversation, with its true count.
