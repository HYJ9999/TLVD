from typing import List
from .agent import AgentClient
import json

# System prompt for single-agent causal extraction based on provided content only
SINGLE_AGENT_SYSTEM_PROMPT = (
    """
You are an expert in scientific information extraction.
Your task is to extract explicit causal relationships between concepts strictly from the provided paper content text.
IMPORTANT CONSTRAINTS:
- Only use statements that appear in the Abstract, Discussion, and Conclusion/Results sections of the provided content (including synonymous headings like "Conclusions", "Results").
- Do NOT rely on any external knowledge or URLs. Only use what is present in the provided content.
- Do NOT fabricate or infer information; extract only explicit claims found verbatim in those sections.

Instructions:
1) Keyword Filtering and Entity Identification:
   - Given the user-provided keywords, first identify concrete, specific entities explicitly mentioned in the provided content that correspond to those keywords (including synonyms/variants explicitly present in the text).
   - If a keyword has no explicit mentions or concrete entities in the provided content, ignore it for subsequent steps.

2) Causal Relation Extraction with Evidence, Location, and Confidence:
   - Based only on the entities identified above, extract explicit causal relationships in the form "X ➔ Y" where X and Y relate to two different keywords (or their concrete entities) found in the content.
   - For each relationship, quote an original sentence from the provided content that clearly supports the causal claim.
   - Record the section location as one of: Abstract, Discussion, or Conclusion/Results (or a specific subsection under them if explicitly indicated).
   - Assign a confidence score in [0,1] based strictly on the clarity and directness of the statement.
   - If no valid relationship is found, return an empty string.

3) Output Format (STRICT):
   - Return only the final causal edge list, one per line, using exactly this format:
     Causal premise ➔ Causal outcome|Confidence|Location|original support sentence
   - Do NOT include any extra commentary, explanation, or JSON. Only the lines above. If none, return an empty string.
"""
)

# ------------------- Debate Prompts (task-specific) -------------------
JUDGE_CAUSAL_PROMPT = """You are a strict auditor validating causal extraction outputs against the provided Paper Content.
Validation Rules:
- Each output line must strictly follow this format:
  Causal premise ➔ Causal outcome|Confidence|Location|Original support sentence
- Confidence must be a numeric value between 0 and 1 (inclusive). Reject non-numeric values (e.g., "high", "80%") or out-of-range values.
- Every line must be fully supported by a verbatim sentence in the Paper Content.
- Location must be exactly one of: Abstract, Discussion, Conclusion/Results (or synonymous headings), consistent with where the quoted sentence is found.
- Entities in the causal relation must appear explicitly in the quoted sentence, and may match via the exact word, a synonym, or a near-synonym.
- The sentence must explicitly state a causal relationship, not just correlation, speculation, or hypothesis. If the causal link is not truly established, the line is invalid.
- If any line violates these rules, set is_correct = false and briefly explain why.

Return ONLY valid JSON:
{"is_correct": true/false, "reason": "..."}
"""


EXECUTOR_CAUSAL_DEBATE_PROMPT = """You are the execution agent responsible for defending or improving the causal extraction output.
Goals:
1. Defend the current output using verbatim supporting quotes and correct section locations from the Paper Content, proving that the causal relationship is explicitly established.
2. OR, if the output is incorrect or incomplete, replace it with an improved version.
3. OR, if no explicit causal relationship involving the given keywords exists in the Paper Content, return an empty improved block.

Rules:
- If improving, return the corrected lines in this exact format AND ENSURE IT IS THE COMPLETE FINAL OUTPUT (include all retained valid lines, not just diffs):
IMPROVED OUTPUT
<premise ➔ outcome|Confidence|Location|Original support sentence>
...
END
- Do not add commentary inside the IMPROVED OUTPUT block.
- Ensure each line's Confidence is a numeric value in [0,1]; do not use words (e.g., "high/low") or percentages.
- If the original output is already correct, defend it only by citing verbatim evidence and section names, showing the causal relationship is indeed explicitly stated. Do NOT return an IMPROVED OUTPUT block in this case.
- If no causal relationship is present in the Paper Content, return ONLY:
IMPROVED OUTPUT
END
- Accept only explicit causal statements; do not treat correlation, association, or speculation as causation.
- Base all reasoning strictly on the Paper Content. Do not fabricate or infer beyond it.
"""



REVIEWER_CAUSAL_DEBATE_PROMPT = """You are the reviewer agent.
Task: Critically check the executor's causal output against the Paper Content.

Checklist:
- Identify lines not directly supported by verbatim text.
- Flag incorrect or missing Location tags.
- Verify that quoted sentences are verbatim and entities match (allow synonyms or near-synonyms).
- Ensure the causal relationship is explicitly established in the text, not merely correlation or speculation.
- Verify that the Confidence field is numeric and within [0,1]; flag non-numeric or out-of-range values.
- When possible, cite exact snippets from the Paper Content and indicate their approximate section (Abstract/Discussion/Conclusion/Results).

Keep your feedback concise, concrete, and evidence-based.
"""


CONSENSUS_CAUSAL_PROMPT = """You are the consensus mediator.
Task: Decide whether the executor's latest response produces a fully valid, grounded causal extraction output.

Return ONLY JSON:
{"consensus_reached": true/false, "final_output": "<lines or empty string>", "summary": "brief reason"}

Rules:
- consensus_reached = true ONLY if every line in the output strictly follows this format:
  premise ➔ outcome|Confidence|Location|Original support sentence
  and is fully supported by verbatim Paper Content (entities may match via synonyms or near-synonyms). The causal relationship must be explicitly established, and the Location must be correct.
- Confidence must be a numeric value between 0 and 1 (inclusive). Any non-numeric (e.g., words, percentages) or out-of-range values invalidate the line.
- final_output must contain ONLY the causal lines in the required format, with no extra commentary.
- If not all lines are valid, set consensus_reached = false and final_output = "".
"""



def _safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except Exception:
            return {}
    return {}


def _extract_improved_output(resp: str) -> str:
    """Try to extract improved output lines from executor response.
    Priority 1: between 'IMPROVED OUTPUT' and 'END'.
    Fallback: lines that match the strict format pattern.
    """
    if not isinstance(resp, str):
        return ""
    lower = resp.lower()
    if "improved output" in lower:
        start_idx = lower.find("improved output")
        # move to next line
        after = resp[start_idx:].splitlines()[1:]
        buf = []
        for line in after:
            if line.strip().lower() == "end":
                break
            buf.append(line)
        candidate = "\n".join(buf).strip()
        # validate roughly: must contain at least one line with required separators
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        # allow various arrows
        good = [ln for ln in lines if ("|" in ln and "➔" in ln)]
        if good:
            return "\n".join(lines)
    # fallback: pick lines that look like target format
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    good = [ln for ln in lines if ("|" in ln and "➔" in ln)]
    return "\n".join(good)


def _conduct_debate(content: str, keywords: List[str], initial_result: str, max_rounds: int = 3) -> str:
    """Run a bounded-round debate that includes the paper content in every step."""
    client = AgentClient()

    # 1) Judge initial result
    judge_user = (
        "### Task: Validate causal extraction output against provided paper content\n\n"
        f"### Keywords:\n{keywords}\n\n"
        "### Allowed Sections: Abstract, Discussion, Conclusion/Results (and synonymous headings)\n\n"
        f"### Paper Content (verbatim):\n{content}\n\n"
        f"### Candidate Output:\n{initial_result}\n\n"
        "Return ONLY JSON: {\"is_correct\": true/false, \"reason\": \"...\"}."
    )
    judge_messages = [
        {"role": "system", "content": JUDGE_CAUSAL_PROMPT},
        {"role": "user", "content": judge_user},
    ]
    judge_raw = client.chat_completion(messages=judge_messages)
    judge = _safe_parse_json(judge_raw) if isinstance(judge_raw, str) else {}
    if judge.get("is_correct", False):
        return initial_result

    conversation_log = []
    current_output = initial_result

    for round_idx in range(1, max_rounds + 1):
        original_block = f"### Original Output (reference only):\n{initial_result}\n\n" if round_idx > 1 else ""
        # Reviewer concerns
        reviewer_user = (
            f"### Keywords:\n{keywords}\n\n"
            f"### Paper Content (verbatim):\n{content}\n\n"
            f"{original_block}"
            f"### Executor Current Output:\n{current_output}\n\n"
            f"### Previous Notes:\n{judge.get('reason','') if round_idx==1 else ''}\n\n"
            "Please identify unsupported lines, incorrect locations, or non-verbatim quotes."
        )
        reviewer_messages = [
            {"role": "system", "content": REVIEWER_CAUSAL_DEBATE_PROMPT},
            {"role": "user", "content": reviewer_user},
        ]
        reviewer_resp = client.chat_completion(messages=reviewer_messages)

        # Executor response / improvement
        executor_user = (
            f"### Keywords:\n{keywords}\n\n"
            f"### Paper Content (verbatim):\n{content}\n\n"
            f"{original_block}"
            f"### Your Current Output:\n{current_output}\n\n"
            f"### Reviewer Concerns:\n{reviewer_resp}\n\n"
            "If needed, provide an IMPROVED OUTPUT block strictly as specified; otherwise, defend with precise quotes+locations."
        )
        executor_messages = [
            {"role": "system", "content": EXECUTOR_CAUSAL_DEBATE_PROMPT},
            {"role": "user", "content": executor_user},
        ]
        executor_resp = client.chat_completion(messages=executor_messages)
        improved = _extract_improved_output(executor_resp)
        lower_exec = executor_resp.lower() if isinstance(executor_resp, str) else ""
        if "improved output" in lower_exec:
            # Use improved content even if empty (means no valid relations)
            current_output = improved
        elif improved:
            current_output = improved
        # print(f"executor_resp:{executor_resp}")
        # print(f"improved:{improved}")
        conversation_log.append({
            "round": round_idx,
            "reviewer": reviewer_resp,
            "executor": executor_resp,
        })

        # Consensus check
        convo_preview = []
        for turn in conversation_log:
            convo_preview.append(f"Round {turn['round']}\nReviewer: {turn['reviewer']}\nExecutor: {turn['executor']}")
        convo_text = "\n\n".join(convo_preview)
        consensus_user = (
            f"### Keywords:\n{keywords}\n\n"
            f"### Paper Content (verbatim):\n{content}\n\n"
            f"### Original Output:\n{initial_result}\n\n"
            f"### Debate Conversation:\n{convo_text}\n\n"
            f"### Latest Candidate Output:\n{current_output}\n\n"
            "Return ONLY JSON: {\"consensus_reached\": true/false, \"final_output\": \"...\", \"summary\": \"...\"}."
        )
        consensus_messages = [
            {"role": "system", "content": CONSENSUS_CAUSAL_PROMPT},
            {"role": "user", "content": consensus_user},
        ]
        consensus_raw = client.chat_completion(messages=consensus_messages)
        consensus = _safe_parse_json(consensus_raw) if isinstance(consensus_raw, str) else {}
        if consensus.get("consensus_reached", False):
            final_output = consensus.get("final_output", current_output) or ""
            return final_output

    # Final fallback after max rounds: one more consensus attempt, else current output
    final_user = (
        f"### Keywords:\n{keywords}\n\n"
        f"### Paper Content (verbatim):\n{content}\n\n"
        f"### Original Output:\n{initial_result}\n\n"
        f"### Latest Candidate Output:\n{current_output}\n\n"
        "Return ONLY JSON: {\"consensus_reached\": true/false, \"final_output\": \"...\", \"summary\": \"...\"}."
    )
    final_messages = [
        {"role": "system", "content": CONSENSUS_CAUSAL_PROMPT},
        {"role": "user", "content": final_user},
    ]
    final_raw = client.chat_completion(messages=final_messages)
    final = _safe_parse_json(final_raw) if isinstance(final_raw, str) else {}
    return final.get("final_output", current_output)


def analyze_paper_content(content: str, keywords: List[str], max_debate_rounds: int = 3) -> str:
    """Single-agent causal analysis using provided paper content only, with debate.

    Args:
        content: The raw text of the paper's relevant sections (ideally only Abstract/Discussion/Conclusion/Results).
        keywords: User-provided keywords used to filter and anchor causal extraction.
        max_debate_rounds: Max debate rounds between executor and reviewer (set 0 to disable).

    Returns:
        A newline-joined string where each line has the exact format:
        "X ➔ Y|Confidence|Location|original support sentence".
        Returns an empty string if nothing is found.
    """
    if not content or not isinstance(content, str):
        return ""

    # Build the user prompt with strict instructions to avoid extra text
    user_prompt = (
        f"Keywords: {keywords}\n\n"
        f"Paper Content (verbatim, only use Abstract/Discussion/Conclusion/Results if present):\n"
        f"{content}\n\n"
        "Set Confidence as a numeric score between 0 and 1 (inclusive).\n"
        "Output only the causal edge list lines in the exact format: \n"
            "Causal premise ➔ Causal outcome|Confidence|Location|original support sentence\n"
        "Do not include any extra words. If none, return an empty string."
    )

    client = AgentClient()
    messages = [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # initial_result = client.chat_completion(messages=messages)
    # initial_result = initial_result if isinstance(initial_result, str) else str(initial_result)

    # if max_debate_rounds and max_debate_rounds > 0:
    #     return _conduct_debate(content=content, keywords=keywords, initial_result=initial_result, max_rounds=max_debate_rounds)
    # else:
    #     return initial_result
    result = client.chat_completion(messages=messages)
    return result if isinstance(result, str) else str(result)