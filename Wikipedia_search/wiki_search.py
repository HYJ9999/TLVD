from openai import OpenAI
import wikipedia
import json
import re
from paper_search.token_counter import add_usage, write_total_to_results


MODEL: str = "kimi-k2-0711-preview"
API_KEY: str = ""
BASE_URL: str = "https://api.moonshot.cn/v1"


JUDGE_PROMPT = (
    "You are an independent reviewer. You must evaluate the candidate output strictly against the provided Wikipedia context only. "
    "Do not use any external knowledge. If the context does not contain sufficient evidence, the output must be judged incorrect. "
    "When judging correctness, cross-check claims against the provided context and ensure that the causal direction matches. "
    "Return ONLY a single JSON object with keys: is_correct (boolean), reason (string), fixed_output (string). "
    "Do not include any extra text."
)

EXECUTOR_DEBATE_PROMPT = (
    "You are the executing agent. Address the reviewer's concerns using ONLY the provided Wikipedia Context below. "
    "Quote or reference exact sentences or specific sections from the context to support your claims. "
    "When making claims, you MUST include specific citations in the format: [Segment X] or [Article: Title]. "
    "If the previous output was flawed, provide a corrected/improved output grounded in the context. "
    "Keep reasoning concise and avoid any hallucinations or external knowledge. "
    "Your response MUST adhere to the Required Output Format: Start with a single line saying either 'Yes, {A} causes {B}. The context provides strong evidence:' or 'No, the provided context does not support that {A} causes {B}.' Then provide a numbered list of evidence items. Each item should have a short bolded label, followed by a verbatim quote in double quotes and a valid citation immediately after the quote in the form [Segment N] (and optionally [Article: Title]). Provide one brief explanatory sentence grounded in the quoted content. Do not add sections outside this format."
)

REVIEWER_DEBATE_PROMPT = (
    "You are the reviewer in a debate. Critique the executor's response STRICTLY against the provided Wikipedia Context. "
    "Reject any claims not supported by the context. Verify that all citations reference actual content in the context. "
    "Ensure the response adheres to the Required Output Format (Yes/No header + numbered evidence bullets with verbatim quotes and [Segment N]). "
    "If the response resolves your concerns and is fully grounded with proper citations, explicitly acknowledge consensus."
)

CONSENSUS_PROMPT = (
    "You are a neutral mediator. Given the debate transcript and the original output, decide whether consensus is reached. "
    "If yes, produce the final agreed output with proper citations, and ensure it EXACTLY follows the Required Output Format: "
    "Start with a single line saying either 'Yes, A causes B. The context provides strong evidence:' or 'No, the provided context does not support that A causes B.' "
    "Then provide a numbered list of evidence items with a short bolded label, a verbatim quote in double quotes, and a valid citation [Segment N] (and optionally [Article: Title]) immediately after the quote. Keep explanations concise and grounded. "
    "Return ONLY JSON with keys: consensus_reached (boolean), final_output (string)."
)


def gpt_chat(messages, max_tokens=700, temperature=0.3):
    if not API_KEY:
        raise RuntimeError("Missing MOONSHOT_API_KEY in environment for Wikipedia LLM calls.")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Add token usage to global counter
        try:
            add_usage(getattr(resp, "usage", None))
        except Exception:
            pass
        return resp.choices[0].message.content.strip()
    finally:

        try:
            client.close()
        except Exception:
            pass


def safe_parse_json(text: str):
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


def format_conversation(conversation_log):
    lines = []
    for entry in conversation_log:
        lines.append(f"Round {entry['round']}:")
        lines.append(f"Reviewer: {entry['reviewer']}")
        lines.append(f"Executor: {entry['executor']}")
        if entry.get('consensus'):
            lines.append(">>> Consensus reached <<<")
        lines.append("")
    return "\n".join(lines)


def segment_wiki_content(content, max_segment_length=1500):
    """
    Segmenting Wikipedia content by paragraph and character length for precise citation
    """
    if not content:
        return []
    
   
    paragraphs = content.split('\n\n')
    segments = []
    current_segment = ""
    segment_index = 1
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        
        if len(current_segment) + len(para) > max_segment_length and current_segment:
            segments.append({
                "index": segment_index,
                "content": current_segment.strip()
            })
            segment_index += 1
            current_segment = para
        else:
            if current_segment:
                current_segment += "\n\n" + para
            else:
                current_segment = para
    
    
    if current_segment:
        segments.append({
            "index": segment_index,
            "content": current_segment.strip()
        })
    
    return segments


def build_wiki_text_with_segments(results):

    if not results:
        return "", []
    
    all_segments = []
    full_text_parts = []
    global_segment_index = 1
    
    for item in results:
        title = item.get("title", "Untitled")
        
        if "content" in item and item["content"]:
           
            article_segments = segment_wiki_content(item["content"])
            
          
            full_text_parts.append(f"# {title}")
            
            for seg in article_segments:
                segment_text = f"[Segment {global_segment_index}] {seg['content']}"
                full_text_parts.append(segment_text)
                
                all_segments.append({
                    "index": global_segment_index,
                    "title": title,
                    "content": seg["content"]
                })
                global_segment_index += 1
                
        elif "disambiguation" in item:
            opts = "\n".join(f"- {o}" for o in item["disambiguation"])
            content = f"# {title} (Disambiguation)\n{opts}"
            full_text_parts.append(content)
            all_segments.append({
                "index": global_segment_index,
                "title": title,
                "content": opts
            })
            global_segment_index += 1
            
        elif "error" in item:
            content = f"# {title} (Error)\n{item['error']}"
            full_text_parts.append(content)
            all_segments.append({
                "index": global_segment_index,
                "title": title,
                "content": item['error']
            })
            global_segment_index += 1
            
        else:
            content = f"# {title}\n(No content)"
            full_text_parts.append(content)
            all_segments.append({
                "index": global_segment_index,
                "title": title,
                "content": "(No content)"
            })
            global_segment_index += 1
    
    return "\n\n".join(full_text_parts), all_segments


def build_wiki_text(results):

    if not results:
        return ""
    chunks = []
    for item in results:
        title = item.get("title", "Untitled")
        if "content" in item and item["content"]:
            chunks.append(f"# {title}\n{item['content']}")
        elif "disambiguation" in item:
            opts = "\n".join(f"- {o}" for o in item["disambiguation"])
            chunks.append(f"# {title} (Disambiguation)\n{opts}")
        elif "error" in item:
            chunks.append(f"# {title} (Error)\n{item['error']}")
        else:
            chunks.append(f"# {title}\n(No content)")
    return "\n\n".join(chunks)


# Query Wikipedia using API
def query_wikipedia(query, lang='en', max_results=3):
    wikipedia.set_lang(lang)
    results = []
    try:

        search_results = wikipedia.search(query, results=max_results)
        if not search_results:
            return None
        
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                results.append({
                    "title": page.title,
                    "content": page.content
                })
            except wikipedia.exceptions.DisambiguationError as e:

                results.append({
                    "title": title,
                    "disambiguation": e.options[:max_results]
                })
            except Exception as e:
                results.append({
                    "title": title,
                    "error": str(e)
                })

        return results
    
    except wikipedia.exceptions.HTTPTimeoutError:
        print("HTTP Timeout Error: Please try again later.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# Use LLM to Check for Causal Relationship (executor)
def check_causal_relationship_with_gpt(text, keyword_a, keyword_b):
    
    if isinstance(text, list):
        text = build_wiki_text(text)

    prompt = f"""
You are assessing whether a causal relationship exists between two concepts using ONLY the provided Wikipedia context.

Query: {keyword_a} and {keyword_b}

Context:
{text}

Task: Based on the context, determine whether {keyword_a} causes or leads to {keyword_b}. 
If yes, quote exact sentences from the context and include specific citations in the format [Segment X] or [Article: Title]. 
Briefly explain the mechanism or evidence. 
If no, explain why not or whether the context is insufficient. 
Keep your answer concise, factual, and properly cited.

Required Output Format:
- Start with one line:
  - If supported: "Yes, {keyword_a} causes {keyword_b}. The context provides strong evidence:"
  - If not supported: "No, the provided context does not support that {keyword_a} causes {keyword_b}."
- Then provide a numbered list of evidence items. For each item:
  1. Provide a short bolded label (e.g., **Direct causation**).
  2. Provide a verbatim quote in double quotes, immediately followed by a valid citation [Segment N] (and optionally [Article: Title]).
  3. Add one concise explanatory sentence grounded in the quoted content.
- Do not include any content not grounded in the context. Do not use external knowledge. Do not add extra sections outside this format.
"""


    try:
        answer = gpt_chat([{"role": "user", "content": prompt}], max_tokens=600, temperature=0.3)
        return answer
    except Exception as e:

        print(f"Error occurred: {e}")
        return ""
    


# Debate: reviewer, executor, mediator consensus
def conduct_debate_for_wiki(keyword_a, keyword_b, wiki_context: str, initial_output: str, max_rounds: int = 3):
    overall_task = (
        "Determine if there is a causal relationship between two concepts using **ONLY** the provided Wikipedia information."
    )
    objective = f"Assess causal relationship: {keyword_a} -> {keyword_b}"

    # Initial review
    judge_user_content = (
        f"### Overall Task:\n{overall_task}\n\n"
        f"### Subtask Objective:\n{objective}\n\n"
        f"### Context (Wikipedia excerpts):\n{wiki_context}\n\n"
        f"### Candidate Output:\n{initial_output}\n\n"
        f"Please validate the Candidate Output strictly against the Context and reply with a single JSON object as specified."
    )
    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": judge_user_content}
    ]
    judge_raw = gpt_chat(judge_messages, max_tokens=300, temperature=0.1)
    judge = safe_parse_json(judge_raw)

    is_correct = judge.get('is_correct', True)
    judge_reason = judge.get('reason', '')

    conversation_log = []
    if is_correct:
        conversation_log.append({
            'round': 0,
            'reviewer': f"Output is correct: {judge_reason}",
            'executor': "Acknowledged.",
            'consensus': True
        })
        return initial_output, conversation_log, True

    current_output = initial_output

    for round_num in range(1, max_rounds + 1):
        # Reviewer concerns
        if round_num == 1:
            reviewer_concern = f"I have concerns about your output: {judge_reason}"
        else:
            reviewer_messages = [
                {"role": "system", "content": REVIEWER_DEBATE_PROMPT},
                {"role": "user", "content": (
                    f"Overall Task:\n{overall_task}\n\n"
                    f"Wikipedia Context (for strict grounding):\n{wiki_context}\n\n"
                    f"Previous conversation:\n{format_conversation(conversation_log)}\n\n"
                    f"The executor's latest response: {executor_response}\n\n"
                    f"Please evaluate their response with strict reference to the Wikipedia Context and the Required Output Format."
                )}
            ]
            reviewer_concern = gpt_chat(reviewer_messages, max_tokens=300, temperature=0.2)

        # Executor response
        executor_messages = [
            {"role": "system", "content": EXECUTOR_DEBATE_PROMPT},
            {"role": "user", "content": (
                f"### Task Context:\n"
                f"Overall Task: {overall_task}\n"
                f"Subtask: {objective}\n"
                f"Wikipedia Context (use only this):\n{wiki_context}\n\n"
                f"Your Current Output:\n{current_output}\n\n"
                f"### Reviewer's Concerns:\n{reviewer_concern}\n\n"
                f"Please respond to the reviewer's concerns with exact quotes and specific citations from the Wikipedia Context, strictly following the Required Output Format."
            )}
        ]
        executor_response = gpt_chat(executor_messages, max_tokens=600, temperature=0.3)

        conversation_log.append({
            'round': round_num,
            'reviewer': reviewer_concern,
            'executor': executor_response,
            'consensus': False
        })

        # Check consensus
        consensus_messages = [
            {"role": "system", "content": CONSENSUS_PROMPT},
            {"role": "user", "content": (
                f"### Original Task Output:\n{initial_output}\n\n"
                f"### Debate Conversation:\n{format_conversation(conversation_log)}\n\n"
                f"Please determine if consensus has been reached and what the final output should be (the final_output must follow the Required Output Format)."
            )}
        ]
        consensus_raw = gpt_chat(consensus_messages, max_tokens=400, temperature=0.1)
        consensus = safe_parse_json(consensus_raw)

        consensus_reached = consensus.get('consensus_reached', False)
        final_output = consensus.get('final_output', current_output)

        if consensus_reached:
            conversation_log[-1]['consensus'] = True
            return final_output, conversation_log, True

        # Update for next round
        current_output = final_output

    # If no consensus, return last output and conversation log
    return current_output, conversation_log, False


def extract_citations_and_sources(final_output):

    citations = []


    segment_pattern = r'\[Segment (\d+)\]'
    segment_matches = re.findall(segment_pattern, final_output)
    segment_indices = [int(n) for n in segment_matches]

    article_pattern = r'\[Article: ([^\]]+)\]'
    article_matches = re.findall(article_pattern, final_output)

    if segment_matches:
        citations.extend([f"Segment {num}" for num in segment_matches])

    if article_matches:
        citations.extend([f"Article: {title}" for title in article_matches])


    quote_pattern = r'"([^"]+)"'
    quotes = re.findall(quote_pattern, final_output)

    return citations, segment_indices, quotes


def verify_citations_exist(final_output, wiki_context, segments):

    citations, segment_indices, quotes = extract_citations_and_sources(final_output)


    segment_map = {s.get("index"): (s.get("title", "Untitled"), s.get("content", "")) for s in segments}

    pair_pattern = r'["“]([^"”]+)["”]\s*\[Segment\s+(\d+)\]'
    pairs = re.findall(pair_pattern, final_output, flags=re.IGNORECASE | re.DOTALL)

    for qtext, idx_str in pairs:
        try:
            idx = int(idx_str)
        except ValueError:
            return False, f"Malformed segment index in citation: [Segment {idx_str}]"
        if idx not in segment_map:
            return False, f"Referenced [Segment {idx}] does not exist in the provided Wikipedia context."
        _title, seg_content = segment_map[idx]
        if qtext.strip() and qtext.strip() not in seg_content:

            return False, (f"Quoted text not found in cited segment: '...{qtext.strip()[:60]}...' in [Segment {idx}].")

    if segment_indices:
        valid_segments = set(segment_map.keys())
        for idx in segment_indices:
            if idx not in valid_segments:
                return False, f"Referenced [Segment {idx}] does not exist in the provided Wikipedia context."


    if quotes:
        for quote in quotes:
            if quote and quote not in wiki_context:
                return False, f"Quoted text '{quote[:50]}...' does not appear verbatim in the Wikipedia context."

    
    if not segment_indices and not quotes and ("evidence" in final_output.lower() or "support" in final_output.lower()):
        if "[Article:" not in final_output and "[Segment" not in final_output:
            return False, "Claims evidence but provides no verifiable citations from the context."

    return True, "All citations verified successfully."


def find_causal_evidence(query_a, query_b):
    
    """
    High-level orchestration: query Wikipedia, run LLM review, and conduct debate if necessary.
    Enhanced with citation verification - if citations don't exist in original context, re-verify.
    """
    print("-----------------wiki search-----------------")

    # Step 1: get wikipedia content
    wiki_results = query_wikipedia(f"{query_a} {query_b}")


    if not wiki_results:
        return "No sufficient Wikipedia content found for this pair."

    for item in wiki_results:
        print(item["title"])
    # Step 2: build segmented context for better citation
    wiki_context, segments = build_wiki_text_with_segments(wiki_results)

    # Step 3: initial LLM check with enhanced prompting for citations
    final_output = check_causal_relationship_with_gpt(wiki_context, query_a, query_b)
    if not final_output:
        return "No, something error"

    # Step 4: debate (to improve credibility and ensure grounding)
    #final_output, conversation_log, consensus = conduct_debate_for_wiki(query_a, query_b, wiki_context, initial_output)

    # Step 5: verify citations actually exist in the context
    is_valid, reason = verify_citations_exist(final_output, wiki_context, segments)
    
    if not is_valid:
        print(f"Citation validation failed: {reason}")
        print("Re-verifying with cached Wikipedia context...")
        

        re_verify_prompt = f"""
The previous analysis contained citations that don't exist in the provided context or were not verifiable. 
Please re-analyze the causal relationship between {query_a} and {query_b} using ONLY the information below.
Guidelines:
- Prefer including at least one verbatim quote from the context (enclose it in double quotes "...") when possible.
- At minimum, include correct citations like [Segment X] or [Article: Title] that truly exist in the context.
- Do NOT invent content or rely on external knowledge. If no sufficient evidence exists in the context, clearly say so.
- Start with a single line saying either **'Yes, A causes B. The context provides strong evidence:' or 'No, the provided context does not support that A causes B.'**
Context:
{wiki_context}

Question: Does {query_a} cause or lead to {query_b} based on the provided context?
"""
        
        re_verified_output = gpt_chat([
            {"role": "user", "content": re_verify_prompt}
        ], max_tokens=600, temperature=0.3)

        pattern = r"^Yes,\s?.+"
        match = re.match(pattern, re_verified_output)
        if match:
            print("Re-verification successful with valid citations.")
            final_output = re_verified_output
        else:
            print(f"Re-verification still failed")

            final_output = f"No. Based on available Wikipedia content, insufficient reliable evidence found to establish clear causal relationship between {query_a} and {query_b}."    
    # Step 6: extract final citations for reference (but don't append original text)
    citations, _, _ = extract_citations_and_sources(final_output)
    
    if citations:
        return f"{final_output}\n\nCitations: {', '.join(citations)}"
    else:
        return final_output


def main():
    # Example usage: detect causal evidence between 'smoking' and 'lung cancer'
    result = find_causal_evidence("smoking", "lung cancer")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
