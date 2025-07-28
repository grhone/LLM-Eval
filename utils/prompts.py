JUDGE_ANSWER_SYSTEM_PROMPT = """You will be provided with the summary of a document, a piece of text, a question generated from that text, and the correct or "gold" answer to the question. Additionally, you will receive a model answer. Your task is to determine wether the model answer is correct using the provided "gold" answer as a reference.

# Steps

1. **Document Understanding**:
   - Analyze the provided document summary to grasp the context and main themes.

2. **Chunk Understanding**:
   - Examine the provided text (chunk) to understand its content.

3. **Question Understanding**:
   - Interpret the given question to fully comprehend what is being asked.

4. **Ground Truth Answer Understanding**:
   - Understand the provided ground truth answer, identifying its key points.

5. **Answer Understanding**:
   - Examine the Model Answer, identifying key points and assessing accuracy and factuality.

6. **Final Answer**:
   - 0 or 1 (0 if the model answer is incorrect, 1 if it is correct).

# Output Format

- Provide your final evaluation of whether the answer is correct within `<final_answer>` XML tags.
- Include a detailed analysis for each part within the designated XML tags: `<document_understanding>`, `<chunk_understanding>`, `<question_understanding>`, `<ground_truth_answer_understanding>`, `<model_answer_understanding>`, and `<final_answer>`.

# Examples

**Input**:
```xml
<document_summary>
[Summary]
</document_summary>

<piece_of_text>
[Text]
</piece_of_text>

<question>
[Question]
</question>

<gold_answer>
[Gold Answer]
</gold_answer>

<model_answer>
[Model Answer]
</model_answer>
```
**Output**:
```xml

<document_understanding>
Understanding of the summary including key themes
</document_understanding>

<chunk_understanding>
Analysis of the piece of text
</chunk_understanding>

<question_understanding>
Comprehension of the question being asked
</question_understanding>

<ground_truth_answer_understanding>
Key points from the gold answer
</ground_truth_answer_understanding>

<model_answer_understanding>
Key points and accuracy of Answer A
</model_answer_understanding>

<final_answer>
1 or 0 (1 if the model answer is correct, 0 if it is incorrect)
</final_answer>
```

# Notes

- Always focus on key points and factual correctness as per the ground truth.
- Avoid any biases and rely solely on the evidence presented.
- Enclose all evaluations and analyses in the specified XML tags for clarity and structure."""


JUDGE_ANSWER_USER_PROMPT = """<document_summary>
{summary}
</document_summary>

<piece_of_text>
{chunk}
</piece_of_text>

<question>
{question}
</question>

<gold_answer>
{oracle_answer}
</gold_answer>

<model_answer>
{model_answer}
</model_answer>"""


def get_judge_prompt(question: str, answer: str, gold: str, **kwargs):
    """Create formatted chat messages for answer evaluation.

    Args:
        question (str): The question being evaluated.
        answer (str): Model-generated answer to evaluate.
        gold (str): Ground truth (correct) answer.
        **kwargs: Additional optional arguments:
            chunks (str): Relevant text chunks for context.
            documents (str): Document summary for context.

    Returns:
        list: List of chat messages formatted for judge model:
            - System prompt with evaluation instructions
            - User prompt with question/answer context
    """
    chunk = kwargs.get("chunks", "")
    summary = kwargs.get("documents", "")

    prompt = [
        {"role": "system", "content": JUDGE_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_ANSWER_USER_PROMPT.format(
                summary=summary, chunk=chunk, question=question, oracle_answer=gold, model_answer=answer
            ),
        },
    ]

    return prompt


ZEROSHOT_QA_USER_PROMPT = """Answer the following question:

<question>
{question}
</question>

Enclose your full answer in <answer> XML tags. For example:

<answer>
[your answer here]
</answer>"""


def get_zeroshot_qa_prompt(question: str):
    """Create prompt for zero-shot question answering.

    Args:
        question (str): Question to answer.

    Returns:
        str: Formatted prompt instructing model to:
            - Answer the question
            - Enclose response in <answer> XML tags
    """
    return ZEROSHOT_QA_USER_PROMPT.format(question=question)
