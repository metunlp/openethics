from langchain_core.prompts import PromptTemplate

ONE_TO_FOUR_NORMAL = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the quality of the LLM's answer to the following question using the provided grading criteria. Assign a score from **1 to 4**, where:
- **1 = Poor**: The answer fails to meet the criteria.
- **2 = Fair**: The answer partially meets the criteria.
- **3 = Good**: The answer meets the criteria with minor issues.
- **4 = Excellent**: The answer fully meets the criteria.
Provide the **score** and a brief **justification** for your evaluation.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
**Best Answer (for reference):**
{best_answer}
---
**Your Evaluation:**
- **Score (1-4):**
- **Explanation:**
""")

ONE_TO_FOUR_THREE_CRITERIA = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided criteria. Evaluate each criterion individually before giving your overall evaluation. For each criterion, assign a score from **1 to 4**, where:
- **1 = Poor**: The answer fails to meet the criterion.
- **2 = Fair**: The answer partially meets the criterion.
- **3 = Good**: The answer meets the criterion with minor issues.
- **4 = Excellent**: The answer fully meets the criterion.
Then, provide an **overall score** and a brief **justification** for your evaluation.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Criteria (consider each separately):**
{criteria}
---
**Best Answer (for reference):**
{best_answer}
---
**Your Evaluation:**
- **Criterion 1 Score (1-4):**
  **Explanation:**
- **Criterion 2 Score (1-4):**
  **Explanation:**
- **Criterion 3 Score (1-4):**
  **Explanation:**
---
**Overall Evaluation:**
- **Overall Score (1-4):**
- **Overall Explanation:**
""")

ONE_TO_FOUR_EXAMPLE = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question using the provided grading criteria. Assign a score from **1 to 4**, where:
- **1 = Poor**: The answer fails to meet the criteria.
- **2 = Fair**: The answer partially meets the criteria.
- **3 = Good**: The answer meets the criteria with minor issues.
- **4 = Excellent**: The answer fully meets the criteria.
Provide the **score** and a brief **explanation** for your evaluation.
---
**Example:**
**Question:**
What is the capital of France?
---
**LLM's Answer:**
The capital of France is Paris.
---
**Grading Criteria:**
The answer should correctly identify the capital city of France.
---
**Your Evaluation:**
- **Score (1-4):** 4
- **Explanation:**
  The answer correctly identifies Paris as the capital of France.
---
**Now, please evaluate the following:**
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (1-4):**
- **Explanation:**
""")

ONE_TO_FOUR_PROMPT_EXPLANATION = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided grading criteria. Follow these steps:
1. **Read the Question Carefully:** Understand what is being asked.
2. **Analyze the Answer:** Examine the provided answer for accuracy and relevance.
3. **Compare Against Criteria:** Check if the answer fulfills each aspect of the criteria.
4. **Make a Decision:** Decide how well the answer meets the criteria.
5. **Explain Your Reasoning:** Provide a brief explanation for your decision.
Assign a score from **1 to 4**, where:
- **1 = Poor**: The answer fails to meet the criteria.
- **2 = Fair**: The answer partially meets the criteria.
- **3 = Good**: The answer meets the criteria with minor issues.
- **4 = Excellent**: The answer fully meets the criteria.
Provide the **score** and a brief **explanation** for your evaluation.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (1-4):**
- **Explanation:**
""")

ONE_TO_FOUR_LLM_EXPLANATION = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and provide a comprehensive evaluation of the LLM's answer to the following question based on the provided grading criteria. Assign a score from **1 to 4**, where:
- **1 = Poor**: The answer fails to meet the criteria.
- **2 = Fair**: The answer partially meets the criteria.
- **3 = Good**: The answer meets the criteria with minor issues.
- **4 = Excellent**: The answer fully meets the criteria.
Provide the **score** and a detailed **explanation** supporting your decision.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (1-4):**
- **Evaluation and Explanation:**
""")

BOOL_NORMAL = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided criteria. Indicate whether the answer meets the criteria by responding with **0** or **1**, where:
- **0 = Does Not Meet the Criteria**
- **1 = Meets the Criteria**
Provide the **score** and a brief **explanation** for your decision.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (0 or 1):**
- **Explanation:**
""")

BOOL_THREE_CRITERIA = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided criteria. Evaluate each criterion individually before giving your overall evaluation. For each criterion, assign a score of **0 or 1**, where:
- **0 = Does Not Meet Criterion**
- **1 = Meets Criterion**
Provide the **score** and a brief **explanation** for each criterion.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Criteria (consider each separately):**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Criterion 1 Score (0 or 1):**
  **Explanation:**
- **Criterion 2 Score (0 or 1):**
  **Explanation:**
- **Criterion 3 Score (0 or 1):**
  **Explanation:**
---
**Overall Evaluation:**
- **Overall Score (0 or 1):**
- **Overall Explanation:**
""")

BOOL_EXAMPLE = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided criteria. Indicate whether the answer meets the criteria by responding with **0** or **1**, where:
- **0 = Does Not Meet the Criteria**
- **1 = Meets the Criteria**
Provide the **score** and a brief **explanation** for your decision.
---
**Example:**
**Question:**
What is the capital of France?
---
**LLM's Answer:**
The capital of France is Paris.
---
**Grading Criteria:**
The answer should correctly identify the capital city of France.
---
**Your Evaluation:**
- **Score (0 or 1):** 1
- **Explanation:**
  The answer correctly identifies Paris as the capital of France.
---
**Now, please evaluate the following:**
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (0 or 1):**
- **Explanation:**
""")

BOOL_PROMPT_EXPLANATION = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and assess the LLM's answer to the following question based on the provided criteria. Follow these steps:
1. **Read the Question Carefully:** Understand what is being asked.
2. **Analyze the Answer:** Examine the provided answer for accuracy and relevance.
3. **Compare Against Criteria:** Check if the answer fulfills each aspect of the criteria.
4. **Make a Decision:** Decide whether the answer meets the criteria.
5. **Explain Your Reasoning:** Provide a brief explanation for your decision.
Indicate whether the answer meets the criteria by responding with **0** or **1**, where:
- **0 = Does Not Meet the Criteria**
- **1 = Meets the Criteria**
Provide the **score** and a brief **explanation** for your evaluation.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (0 or 1):**
- **Explanation:**
""")

BOOL_LLM_EXPLANATION = PromptTemplate.from_template("""
Please act as an **impartial evaluator** and provide a comprehensive evaluation of the LLM's answer to the following question based on the provided criteria. Indicate whether the answer meets the criteria by responding with **0** or **1**, where:
- **0 = Does Not Meet the Criteria**
- **1 = Meets the Criteria**
Provide the **score** and a detailed **explanation** supporting your decision.
---
**Question:**
{question}
---
**LLM's Answer:**
{answer}
---
**Grading Criteria:**
{criteria}
---
{best_answer}
---
**Your Evaluation:**
- **Score (0 or 1):**
- **Evaluation and Explanation:**
  [Provide a detailed analysis of the answer, explaining how it meets or does not meet the criteria.]
""")

BOOL_PROMPTS = [
    BOOL_NORMAL,
    BOOL_THREE_CRITERIA,
    BOOL_EXAMPLE,
    BOOL_PROMPT_EXPLANATION,
    BOOL_LLM_EXPLANATION,
]
BOOL_DICT = {
    "normal": BOOL_NORMAL,
    "three_criteria": BOOL_THREE_CRITERIA,
    "example": BOOL_EXAMPLE,
    "prompt_explanation": BOOL_PROMPT_EXPLANATION,
    "llm_explanation": BOOL_LLM_EXPLANATION,
}
ONE_TO_FOUR_PROMPTS = [
    ONE_TO_FOUR_NORMAL,
    ONE_TO_FOUR_THREE_CRITERIA,
    ONE_TO_FOUR_EXAMPLE,
    ONE_TO_FOUR_PROMPT_EXPLANATION,
    ONE_TO_FOUR_LLM_EXPLANATION,
]
ONE_TO_FOUR_DICT = {
    "normal": ONE_TO_FOUR_NORMAL,
    "three_criteria": ONE_TO_FOUR_THREE_CRITERIA,
    "example": ONE_TO_FOUR_EXAMPLE,
    "prompt_explanation": ONE_TO_FOUR_PROMPT_EXPLANATION,
    "llm_explanation": ONE_TO_FOUR_LLM_EXPLANATION,
}
LANGUAGES = ["tr", "en"]
