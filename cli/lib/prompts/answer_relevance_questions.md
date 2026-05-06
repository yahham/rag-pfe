Given the following answer, generate {n_questions} distinct questions that
this answer could plausibly be responding to.

Answer: {answer}

Instructions:
- Each question should be self-contained and specific
- The questions should reflect what someone asking this answer's topic would ask
- Do not copy phrases from the answer verbatim
- If the answer is vague or uninformative, generate appropriately broad questions

Return ONLY a valid JSON list of {n_questions} question strings.
No explanation, no markdown, no extra text.

Example format: ["question one?", "question two?", "question three?"]
