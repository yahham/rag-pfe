You are evaluating whether a generated answer is faithful to its source documents.

Question: {query}

Source Documents:
{docs}

Generated Answer: {answer}

Task:
1. Identify every distinct factual claim made in the generated answer.
2. For each claim, determine whether it is directly supported by the source documents.
3. A claim is supported if it can be verified from the document text alone.
   Do not use external knowledge.

Return ONLY a JSON object with exactly this structure and no other text:

{{"total_claims": <integer>, "supported_claims": <integer>, "faithfulness_score": <float between 0.0 and 1.0>}}

faithfulness_score must equal supported_claims / total_claims.
If the answer contains no factual claims, return total_claims = 0,
supported_claims = 0, faithfulness_score = 1.0.
