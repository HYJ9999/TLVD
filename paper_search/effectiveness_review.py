
from paper_search.agent import AgentClient

def validate_evidence_url(paper_url, analysis_text, keywords):
    """
    Ask LLM to strictly verify whether at least one claimed causal edge (from analysis_text)
    truly exists in the provided paper URL within Abstract/Discussion/Conclusion sections,
    and that the quoted evidence sentence appears in the stated section and matches the edge direction.
    Returns raw LLM response (expected to be 'yes' or 'no').
    """
    system_prompt = """You are a strict scientific paper auditor.
Task:
Given a paper URL and the claimed causal analysis lines produced by an external workflow, you must strictly verify the following ONLY within the Abstract, Discussion, and Conclusion sections of the paper:
- Whether at least one causal relationship line is correct in all aspects:
  1) The quoted evidence sentence exists verbatim in the specified section;
  2) The section label (Location) is indeed one of Abstract, Discussion, or Conclusion and actually matches where the sentence appears;
  3) The sentence clearly supports the causal direction between two entities that correspond to the provided keywords (allowing synonyms/variants explicitly present in the paper);
  4) No content from other sections may be used.
If at least one line passes all checks, output exactly: yes
Otherwise, output exactly: no
Do not include any explanations or extra text.
"""
    user_query = f"""
Paper URL: {paper_url}
Keywords: {keywords}
Claimed analysis lines (may include multiple lines, each in the format `Cause ➔ Effect|Confidence|Location|original support sentence`):
{analysis_text}
"""
    client = AgentClient()
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    result = client.chat_completion(messages=prompt)
    return result

def validate_evidence_by_content(content: str, analysis_text: str, keywords):
    """Use LLM to verify claimed causal edges strictly within Abstract/Discussion/Conclusion
    using provided paper content (no URL). Return raw LLM response ('yes' or 'no').
    """
    if not content or not isinstance(content, str) or not content.strip():
        return "no"
    system_prompt = (
        "You are a strict scientific paper auditor.\n"
        "Task: Given a paper's content and the claimed causal analysis lines, strictly verify the following ONLY within the Abstract, Discussion, and Conclusion sections of the content:\n"
        "- Whether at least one causal relationship line is correct in all aspects:\n"
        "  1) The quoted evidence sentence exists verbatim in the provided content;\n"
        "  2) The section label (Location) is indeed one of Abstract, Discussion, or Conclusion and actually matches where the sentence appears in the content;\n"
        "  3) The sentence clearly supports the causal relationship between two entities that correspond to the provided keywords (allowing synonyms/variants explicitly present in the content);\n"
        "  4) No content from other sections may be used.\n"
        "If at least one line passes all checks, output exactly: yes\n"
        "Otherwise, output exactly: no\n"
        "Do not include any explanations or extra text."
    )
    user_query = (
        f"Keywords: {keywords}\n"
        f"Claimed analysis lines (format `Cause ➔ Effect|Confidence|Location|original support sentence`):\n{analysis_text}\n\n"
        f"Paper content (only Abstract/Discussion/Conclusion should be considered):\n{content}"
    )
    client = AgentClient()
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    result = client.chat_completion(messages=prompt)
    return result


def paper_effectiveness_review(paper_list, keywords):
    
    aggregated_analysis_text = ""
    valid_paper_count = 0
    total_papers = 0
    validated_urls = []

    if paper_list:
        if isinstance(paper_list, list):
            total_papers = len(paper_list)
            parts = []
            for res in paper_list:
                if isinstance(res, dict):
                    url = res.get("url")
                    content = res.get("content")
                    text = res.get("analysis", "")
                    
                    if content and isinstance(content, str) and content.strip():
                        try:
                            vresp = validate_evidence_by_content(content=content, analysis_text=text, keywords=keywords)
                            if isinstance(vresp, str) and 'yes' in vresp.strip().lower():
                                valid_paper_count += 1
                                if url:
                                    validated_urls.append(url)
                                parts.append(f"url: {url}\n{text}" if url else text)
                            else:
                                
                                try:
                                    res["analysis"] = ""
                                except Exception:
                                    pass
                                parts.append(f"url: {url}\n[No valid evidence found]" if url else "[No valid evidence found]")
                        except Exception as e:
                            print(f"validate_evidence_by_content error for entry: {e}")
                            try:
                                res["analysis"] = ""
                            except Exception:
                                pass
                            parts.append(f"url: {url}\n[No valid evidence found]" if url else "[No valid evidence found]")
                    elif url:
                        try:
                            vresp = validate_evidence_url(paper_url=url, analysis_text=text, keywords=keywords)
                            if isinstance(vresp, str) and 'yes' in vresp.strip().lower():
                                valid_paper_count += 1
                                validated_urls.append(url)
                                parts.append(f"url: {url}\n{text}")
                            else:
                                
                                try:
                                    res["analysis"] = ""
                                except Exception:
                                    pass
                                parts.append(f"url: {url}\n[No valid evidence found]")
                        except Exception as e:
                            print(f"validate_evidence error for {url}: {e}")
                            
                            try:
                                res["analysis"] = ""
                            except Exception:
                                pass
                            parts.append(f"url: {url}\n[No valid evidence found]")
                    else:
                        
                        parts.append(text)
                else:
                    
                    parts.append(str(res))
            aggregated_analysis_text = "\n\n".join(parts)
        else:
            aggregated_analysis_text = str(paper_list)

    print(f"Valid supporting papers for edge {keywords}: {valid_paper_count}/{total_papers}")
    if validated_urls:
        print(f"Validated URLs: {validated_urls}")
    
    validate_info = [
        f"\nValid supporting papers for edge {keywords}: {valid_paper_count}/{total_papers}",
        f"Validated URLs: {validated_urls}\n"
    ]
    count_info = {"valid":valid_paper_count, "total":total_papers}
    
    return aggregated_analysis_text, validate_info, paper_list, count_info