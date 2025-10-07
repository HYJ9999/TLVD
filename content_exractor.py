import os
import sys
import time
import threading
from collections import deque
import requests
import re


_RJINA_MAX_CALLS = 20
_RJINA_WINDOW_SEC = 60
_RJINA_CALL_TIMES = deque()
_RJINA_RATE_LOCK = threading.Lock()

def _rjina_rate_limit_wait(api_base: str) -> None:
    
    if not isinstance(api_base, str) or "r.jina.ai" not in api_base.lower():
        return
    while True:
        now = time.time()
        with _RJINA_RATE_LOCK:
            cutoff = now - _RJINA_WINDOW_SEC
            while _RJINA_CALL_TIMES and _RJINA_CALL_TIMES[0] < cutoff:
                _RJINA_CALL_TIMES.popleft()
            if len(_RJINA_CALL_TIMES) < _RJINA_MAX_CALLS:
                _RJINA_CALL_TIMES.append(now)
                return

            earliest = _RJINA_CALL_TIMES[0]
            sleep_s = max(0.0, earliest + _RJINA_WINDOW_SEC - now) + 0.01
        time.sleep(sleep_s)


def _clean_content_for_llm(text: str) -> str:

    if not isinstance(text, str) or not text:
        return text
    
    s = text
    

    table_fig_pattern = re.compile(
        r'(?:^|\n)\s*(?:table|figure|fig\.?|chart|graph|diagram)\s*\d*[\.:]?\s*[^\n]*\n'
        r'(?:[^\n]*\n){0,20}?'  
        r'(?=\n\s*(?:[A-Z][a-z]|\d+\.|\n|$))', 
        re.IGNORECASE | re.MULTILINE
    )
    s = table_fig_pattern.sub('\n', s)
    

    ascii_table_pattern = re.compile(r'^[|\-+=\s]{10,}$', re.MULTILINE)
    s = ascii_table_pattern.sub('', s)
    

    lines = s.split('\n')
    new_lines = []
    i = 0
    def _is_pipe_table_line(ln: str) -> bool:
        if not isinstance(ln, str):
            return False

        if re.match(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$', ln):
            return True
        if re.match(r'^\s*[-+|=:]{3,}\s*$', ln):
            return True

        if re.match(r'^\s*\|.*\|\s*$', ln):
            return True
        if ln.count('|') >= 2 and re.search(r'\S\|\S', ln):
            return True
        return False
    while i < len(lines):
        if _is_pipe_table_line(lines[i]):
            j = i
            cnt = 0
            while j < len(lines) and _is_pipe_table_line(lines[j]):
                cnt += 1
                j += 1

            if new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            i = j
            continue
        else:
            new_lines.append(lines[i])
            i += 1
    s = '\n'.join(new_lines)


    multi_column_pattern = re.compile(r'^[^\n]*\t{2,}[^\n]*$', re.MULTILINE)
    s = multi_column_pattern.sub('', s)

    image_placeholder_pattern = re.compile(
        r'\[(?:image|img|figure|fig|photo|picture)(?:\s+\d+)?\]|'
        r'<img[^>]*>|'
        r'!\[.*?\]\([^)]*\)|'
        r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|svg|webp)',
        re.IGNORECASE
    )
    s = image_placeholder_pattern.sub('', s)
    

    cite_link_pattern_nested = re.compile(r"\[\[\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?)*\s*\]\]\([^)]*\)\]?")
    s = cite_link_pattern_nested.sub('', s)

    cite_link_pattern = re.compile(r"\[\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?)*\s*\]\([^)]*\)\]?")
    s = cite_link_pattern.sub('', s)

    cite_bracket_pattern = re.compile(r"\[\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-‑–—]\s*\d{1,4})?)*\s*\]")
    s = cite_bracket_pattern.sub('', s)
    
    table_fig_link_md = re.compile(
        r"\(?\[\s*(?:table|tables|figure|fig\.?|figures|supplementary\s+figure|extended\s+data\s+figure)\s*"
        r"(?:[ivxlcdm]+|\d+)?[^\]]*?\]\([^)]*\)\)?",
        re.IGNORECASE
    )
    s = table_fig_link_md.sub('', s)
    

    s = re.sub(r'\n\s*\n\s*\n+', '\n\n', s)

    s = s.replace('\t', ' ')

    s = re.sub(r' {3,}', ' ', s)

    s = re.sub(r'^ +| +$', '', s, flags=re.MULTILINE)
    

    lines = s.split('\n')
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line.strip() != prev_line or len(line.strip()) > 5:  
            cleaned_lines.append(line)
            prev_line = line.strip()
    s = '\n'.join(cleaned_lines)
    

    navigation_pattern = re.compile(
        r'(?:^|\n)\s*(?:navigation|nav|menu|header|footer|sidebar|breadcrumb)[^\n]*\n',
        re.IGNORECASE | re.MULTILINE
    )
    s = navigation_pattern.sub('\n', s)
    

    s = s.strip()
    s = re.sub(r'\n{3,}', '\n\n', s) 
    
    return s


def _trim_after_references(seg: str) -> str:
    if not isinstance(seg, str) or not seg:
        return seg
    s = seg

    ref_pat = re.compile(
        r"^\s*(?:[\-\*\u2022\u2013\u2014\ufeff]*\s*)?(?:[ivxlcdm]+|\d+)?(?:\.\d+)*\s*"
        r"(?:r(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)f(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)r(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)n(?:\s*|-\s*\n\s*|\n\s*)c(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*)s?|bibliograph(?:y|ies)|works\s+cited|literature\s+cited|citations|reference\s+list)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m_ref = ref_pat.search(s)
    if m_ref:
        return s[:m_ref.start()]

    ref_caps_pat = re.compile(
        r"(?:\bREFERENCES\b|\bBIBLIOGRAPH(?:Y|IES)\b|\bWORKS\s+CITED\b|\bLITERATURE\s+CITED\b|\bCITATIONS\b|\bREFERENCE\s+LIST\b)"
    )
    m_caps = ref_caps_pat.search(s)
    if m_caps:
        return s[:m_caps.start()]

    ref_inline_pat = re.compile(
        r"(?:r(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)f(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)r(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*|\n\s*)n(?:\s*|-\s*\n\s*|\n\s*)c(?:\s*|-\s*\n\s*|\n\s*)e(?:\s*|-\s*\n\s*)s?)|\breferences?\b|bibliograph(?:y|ies)|works\s+cited|literature\s+cited|citations|reference\s+list",
        re.IGNORECASE,
    )
    last = None
    for _m in ref_inline_pat.finditer(s):
        last = _m
    if last:
        return s[:last.start()]

    tail_pat = re.compile(
        r"^\s*(?:[\-\*\u2022\u2013\u2014\ufeff]*\s*)?(?:[ivxlcdm]+|\d+)?(?:\.\d+)*\s*"
        r"(?:acknowledg?ments?|appendix(?:es)?|supplementar(?:y|ies)\s+(?:materials?|information)|supplemental\s+materials?|ethics\s+statement|funding|author\s+contributions?|competing\s+interests?|conflicts?\s+of\s+interest|data\s+availability|code\s+availability)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m_tail = tail_pat.search(s)
    if m_tail:
        return s[:m_tail.start()]

    tail_caps_pat = re.compile(
        r"(?:\bACKNOWLEDG?MENTS?\b|\bAPPENDIX(?:ES)?\b|\bSUPPLEMENTAR(?:Y|IES)\s+(?:MATERIALS?|INFORMATION)\b|\bSUPPLEMENTAL\s+MATERIALS?\b|\bETHICS\s+STATEMENT\b|\bFUNDING\b|\bAUTHOR\s+CONTRIBUTIONS?\b|\bCOMPETING\s+INTERESTS?\b|\bCONFLICTS?\s+OF\s+INTEREST\b|\bDATA\s+AVAILABILITY\b|\bCODE\s+AVAILABILITY\b)"
    )
    m_tail_caps = tail_caps_pat.search(s)
    if m_tail_caps:
        return s[:m_tail_caps.start()]
    return s


def _extract_sections_pmc_markdown(md: str) -> dict:
    if not isinstance(md, str) or not md.strip():
        return {"abstract": "", "discussion": "", "conclusion_or_results": ""}

    text = md


    setext_pat = re.compile(r"^(.{1,200})\n[-_]{3,}\s*$", re.MULTILINE)
    matches = list(setext_pat.finditer(text))

    blocks = []  
    if matches:
        for idx, m in enumerate(matches):
            title_line = m.group(1)

            title_line = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", title_line).strip()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            blocks.append((title_line, start, end))
    else:

        atx_pat = re.compile(r"^(?:#|##)\s+(.+?)\s*$", re.MULTILINE)
        atx_matches = list(atx_pat.finditer(text))
        for idx, m in enumerate(atx_matches):
            title_line = m.group(1).strip()
            start = m.end()
            end = atx_matches[idx + 1].start() if idx + 1 < len(atx_matches) else len(text)
            blocks.append((title_line, start, end))

    def norm_title(t: str) -> str:
        t = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", t)  
        t = t.lower()
        t = re.sub(r"^\s*(?:\d+(?:\.\d+)*)\s*", "", t)  
        t = re.sub(r"[^a-z0-9\s/&-]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


    sections_ordered = []  # (key, title, body, start_index)
    for (title, s_idx, e_idx) in blocks:
        body = text[s_idx:e_idx].strip()
        key = norm_title(title)
        sections_ordered.append((key, title.strip(), body, s_idx))

    result = {"abstract": "", "discussion": "", "conclusion_or_results": ""}


    discussion_candidates = []
    conclusion_parts = []
    results_parts = []
    
    for key, title, body, pos in sections_ordered:

        if "abstract" in key and not result["abstract"]:
            result["abstract"] = body

        elif re.search(r"\bdiscussion(s)?\b", key):
            discussion_candidates.append((pos, body))

        elif "conclusion" in key:
            conclusion_parts.append((pos, body))
        elif "result" in key:
            results_parts.append((pos, body))

    if discussion_candidates:
        discussion_candidates.sort(key=lambda x: x[0])
        result["discussion"] = discussion_candidates[0][1]
    

    if conclusion_parts:
        conclusion_parts.sort(key=lambda x: x[0])
        result["conclusion_or_results"] = "\n\n".join([b for _, b in conclusion_parts]).strip()
    elif results_parts:
        results_parts.sort(key=lambda x: x[0])
        result["conclusion_or_results"] = "\n\n".join([b for _, b in results_parts]).strip()


    for k in ("abstract", "discussion", "conclusion_or_results"):
        if result.get(k):
            result[k] = _trim_after_references(result[k])


    for k in list(result.keys()):
        if isinstance(result[k], str) and result[k]:
            result[k] = _clean_content_for_llm(result[k])

    return result


def extract_sections_with_agent(content: str, source: str = "pmc", limit: int = 80000) -> dict:

    
    def _trim_to_limit(text: str, limit: int) -> str:
        if not isinstance(text, str) or limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        
        cut = text.rfind("\n\n", 0, limit)
        if cut < 0:

            candidates = [
                text.rfind("。", 0, limit), text.rfind(".", 0, limit),
                text.rfind("！", 0, limit), text.rfind("!", 0, limit),
                text.rfind("？", 0, limit), text.rfind("?", 0, limit),
            ]
            cut = max(candidates) if any(c >= 0 for c in candidates) else -1
        if cut < 0:
            cut = limit
        return text[:cut].rstrip()

    if source and isinstance(source, str) and source.lower() == "pmc":

        res = _extract_sections_pmc_markdown(content)
    else:
        res = _extract_sections_arxiv_markdown(content) 


    len_abstract = len(res.get("abstract", "") or "")
    
    if len_abstract > limit:

        res["abstract"] = _trim_to_limit(res.get("abstract", ""), limit)
        res["discussion"] = ""
        res["conclusion_or_results"] = ""
    else:

        len_conclusion = len(res.get("conclusion_or_results", "") or "")
        if len_abstract + len_conclusion > limit:
            allowed_for_conclusion = max(0, limit - len_abstract)
            res["conclusion_or_results"] = _trim_to_limit(res.get("conclusion_or_results", ""), allowed_for_conclusion)


            res["discussion"] = ""

        else:

            len_discussion = len(res.get("discussion", "") or "")
            if len_abstract + len_conclusion + len_discussion > limit:
               
                allowed_for_discussion = max(0, limit - len_abstract - len_conclusion)
                res["discussion"] = _trim_to_limit(res.get("discussion", ""), allowed_for_discussion)
    

    return res


def _extract_sections_arxiv_markdown(md: str) -> dict:

    if not isinstance(md, str) or not md.strip():
        return {"abstract": "", "discussion": "", "conclusion_or_results": ""}

    text = md


    lines = text.splitlines(True)  
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    markers: list[tuple[int, int, str]] = []  


    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\r\n")
        h_start = line_starts[i]


        m_atx = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m_atx:
            title_line = m_atx.group(2).strip()
            body_start = h_start + len(raw)
            markers.append((h_start, body_start, title_line))
            i += 1
            continue


        if re.match(r"^(#{1,6})\s*$", line) and i + 1 < len(lines):
            next_title = lines[i + 1].strip()
            if next_title:
                title_line = next_title
                body_start = line_starts[i + 1] + len(lines[i + 1])
                markers.append((h_start, body_start, title_line))
                i += 2
                continue


        if i + 1 < len(lines) and re.match(r"^[-_=]{3,}\s*$", lines[i + 1]):
             
             title_line = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", line).strip()
             if title_line:
                 body_start = line_starts[i + 1] + len(lines[i + 1])
                 markers.append((h_start, body_start, title_line))
                 i += 2
                 continue

        
        m_inline = re.match(
            r"^\s*(?:\d+(?:\.\d+)*)?\s*"
            r"(abstract|introduction|background|methods|materials(?:\s+and\s+methods)?|discussion(?:s)?|results?|conclusions?)"
            r"\s*(?:[:\.-])?\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if m_inline:
            title_kw = m_inline.group(1)
            remainder = m_inline.group(2) or ""
            
            if remainder.strip():

                rel_offset = len(line) - len(remainder)
                body_start = h_start + rel_offset
            else:
                body_start = h_start + len(raw)
            markers.append((h_start, body_start, title_kw))
            i += 1
            continue

        i += 1


    if not markers:
        return {"abstract": "", "discussion": "", "conclusion_or_results": ""}


    markers.sort(key=lambda t: t[0])
    blocks: list[tuple[str, int, int]] = []  
    for idx, (h_start, b_start, title) in enumerate(markers):
        next_h_start = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)

        b_end = max(b_start, next_h_start)
        blocks.append((title, b_start, b_end))

    def norm_title(t: str) -> str:
        t = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", t)  
        t = t.lower()
        t = re.sub(r"^\s*(?:\d+(?:\.\d+)*)\s*", "", t)  
        t = re.sub(r"[^a-z0-9\s/&-]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    sections_ordered = []  # (key, title, body, start_index)
    for (title, s_idx, e_idx) in blocks:
        body = text[s_idx:e_idx].strip()
        key = norm_title(title)
        sections_ordered.append((key, title.strip(), body, s_idx))

    result = {"abstract": "", "discussion": "", "conclusion_or_results": ""}

    discussion_candidates = []
    conclusion_parts = []
    results_parts = []

    for key, title, body, pos in sections_ordered:
        if "abstract" in key and not result["abstract"]:
            result["abstract"] = body
        elif re.search(r"\bdiscussion(s)?\b", key):
            discussion_candidates.append((pos, body))
        elif "conclusion" in key:
            conclusion_parts.append((pos, body))
        elif "result" in key:
            results_parts.append((pos, body))


    if discussion_candidates:
        discussion_candidates.sort(key=lambda x: x[0])
        result["discussion"] = discussion_candidates[0][1]

    if conclusion_parts:
        conclusion_parts.sort(key=lambda x: x[0])
        result["conclusion_or_results"] = "\n\n".join([b for _, b in conclusion_parts]).strip()
    elif results_parts:
        results_parts.sort(key=lambda x: x[0])
        result["conclusion_or_results"] = "\n\n".join([b for _, b in results_parts]).strip()



    for k in ("abstract", "discussion", "conclusion_or_results"):
        if result.get(k):
            result[k] = _trim_after_references(result[k])


    for k in list(result.keys()):
        if isinstance(result[k], str) and result[k]:
            result[k] = _clean_content_for_llm(result[k])

    return result

def url_content_extract(target_url: str, api_base: str = "https://r.jina.ai") -> dict:
    
    """Fetch text via r.jina.ai readability proxy and return a dict: {'title', 'length', 'content'}."""
    _rjina_rate_limit_wait(api_base)
    headers = {
    'X-Respond-With': 'markdown'
    }

    # New behavior: r.jina.ai simply returns the readability text for the target URL.
    endpoint = f"{api_base.rstrip('/')}/{target_url}"
    resp = requests.get(endpoint,headers =headers, timeout=30)
    resp.raise_for_status()
    content = resp.text if isinstance(resp.text, str) else str(resp.content or "")

    return {"title": None, "length": len(content), "content": content}
