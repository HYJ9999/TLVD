from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree as ET
from datetime import datetime
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote
from urllib.request import urlopen, Request
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from content_exractor import url_content_extract, extract_sections_with_agent
from .single_agent_analysis import analyze_paper_content
from .agent import AgentClient
def get_json_with_retry(url, params, max_retries=5, timeout=5, wait=10):
    """Perform a GET request and retry until a valid JSON is returned or the maximum number of retries is reached"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            json_data = response.json()
            return json_data
        except Exception as e:
            print(f"Request failed ({url}) attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(wait)
    return None

def pmc_search(keywords, max_results=10, max_attempts=5):#raw=5,new=3
    query = " ".join(keywords)

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    for attempt in range(max_attempts):
        esearch = get_json_with_retry(
            base + "esearch.fcgi",
            params={
                'db': 'pmc',
                'term': query,
                'retmode': 'json',
                'retmax': max_results,
                'usehistory': 'y'
            }
        )
        if not esearch:
            continue

        esearchresult = esearch.get('esearchresult', {})
        webenv = esearchresult.get('webenv')
        query_key = esearchresult.get('querykey')
        if webenv and query_key:
            break
        else:
            print(f"ESearch validation failed attempt {attempt+1}/{max_attempts}, retrying...")
            time.sleep(5)
    else:
        print("Final ESearch validation failed")
        return []


    for attempt in range(max_attempts):
        summary = get_json_with_retry(
            base + "esummary.fcgi",
            params={
                'db': 'pmc',
                'WebEnv': webenv,
                'query_key': query_key,
                'retmode': 'json',
                'retstart': 0,
                'retmax': max_results
            }
        )
        if not summary:
            continue

        if 'result' in summary and 'uids' in summary['result']:
            break
        else:
            #print(summary)
            print(f"ESummary validation failed attempt {attempt+1}/{max_attempts}, retrying...")
            time.sleep(10)
    else:
        print("Final ESummary validation failed")
        return []


    papers = []
    for uid in summary['result']['uids']:
        try:
            entry = summary['result'][uid]
            paper = {
                'title': entry.get('title', '').strip(),
                'published': entry.get('pubdate', ''),
                'url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{entry['uid']}/",
                'pmc_id': entry['uid']
            }


            paper_url = paper['url']
            response = requests.get(paper_url)
            soup = BeautifulSoup(response.text, 'html.parser')


            article_text = soup.get_text()


            paper['content'] = article_text

            papers.append(paper)
        except Exception as e:
            print(f"Error parsing UID={uid}: {e}")
            continue

    return papers

def arxiv_search(
        keywords: List[str],
        max_results: int = 6,
        sort_by: str = "relevance",
        category: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 10
) -> List[Dict]:
    """
    Args:
        keywords (str): The search keywords (e.g., "quantum computing").
        max_results (int): The number of results to return (default is 5).
        sort_by (str): The sorting method ("relevance" or "submittedDate").
        category (str): The subject category (e.g., "cs.CL").
        max_retries (int): The maximum number of retries.
        timeout (int | float): The request timeout duration (in seconds).

    Returns:
        List[Dict]: A list of structured paper dictionaries.
    """

    encoded_keywords = [f'all:{quote(k)}' for k in keywords]

    query_str = " AND ".join(encoded_keywords)

    query_parts = [query_str]

    
    CATEGORY_MAP = {
    "q-bio": [
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
        "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"
    ]

}

    if category:
        if category in CATEGORY_MAP:

            subcats = CATEGORY_MAP[category]
            cat_query = " OR ".join([f"cat:{c}" for c in subcats])
            query_parts.append(f"({cat_query})")
        else:
            query_parts.append(f"cat:{category}")

    params = {
        "search_query": " AND ".join(query_parts),
        "start": "0",
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": "descending" if sort_by == "submittedDate" else "ascending"
    }


    base_url = "http://export.arxiv.org/api/query?"
    query_str = "&".join([f"{k}={v.replace(' ', '+')}" for k, v in params.items()])
    url = base_url + query_str

    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    raise HTTPError(url, response.status, "HTTP Error", response.headers, None)
                xml_data = response.read()
                break
        except HTTPError as e:
            if e.code == 403:
                print(f"Rate limit triggered, waiting 20 seconds before retry ({attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                raise
        except URLError as e:
            print(f"Network error: {e.reason}, retrying...")
            time.sleep(5)
    else:
        raise Exception("Maximum retries reached, request failed")


    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        try:
            paper = {
                'title': entry.find('atom:title', ns).text.replace("\n","").strip(),
                'published': datetime.strptime(
                    entry.find('atom:published', ns).text,
                    '%Y-%m-%dT%H:%M:%SZ'
                ).strftime('%Y-%m-%d'),
                'url': next(link.attrib['href']
                                for link in entry.findall('atom:link', ns)
                                if link.attrib.get('title') == 'pdf'),
                'arxiv_id': entry.find('atom:id', ns).text.split('/')[-1]
            }
            papers.append(paper)
        except (AttributeError, StopIteration) as e:
            print(f"Error parsing entry: {e}, skipping this paper")

    return papers

def agents_causal_analysis(paper_list, keywords = None, source = "arxiv"):
    """Analyze the causality of each paper with multi-agent
    Args:
        source:"arxiv","pmc"
        
    """
    if keywords == None:
        print("keywords list is empty.")
        return []

    if len(paper_list) == 0:
        print("paper list is empty.")
        return []
    else:
        output_content = []
        structured_results = []  # collect structured results for downstream processing

        # Pre-extract sections content concurrently (using content_exractor)
        EXTRACT_WORKERS = int(os.environ.get("CAUSAL_EXTRACT_CONCURRENCY", "2"))

        def _pre_extract(idx, p):
            url = p.get("url")
            if not url:
                return idx, ""
            try:
                api_base = os.environ.get("EXTRACT_API_BASE", "http://10.65.118.145:3000/")
                data = url_content_extract(url, api_base)
                full_content = data.get("content") if isinstance(data, dict) else ""
                if full_content:
                    sections = extract_sections_with_agent(full_content)
                    sections_text = "\n\n".join([
                    f"=== Abstract ===\n{sections['abstract']}" if sections.get("abstract") else "",
                    f"=== Discussion ===\n{sections['discussion']}" if sections.get("discussion") else "",
                    f"=== Conclusion/Results ===\n{sections['conclusion_or_results']}" if sections.get("conclusion_or_results") else ""
                ]).strip()
                    
                else:
                    sections_text = ""

            except Exception as e:
                print(f"Section extraction failed for {url}: {e}")
                sections_text = ""
            return idx, sections_text

        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
            futures = [executor.submit(_pre_extract, i, p) for i, p in enumerate(paper_list, 1)]
            for fut in as_completed(futures):
                idx, sections_text = fut.result()
                paper_list[idx - 1]["content"] = sections_text
            

        MAX_WORKERS = int(os.environ.get("CAUSAL_ANALYSIS_CONCURRENCY", "10"))
        DEBATE_ROUNDS = int(os.environ.get("CAUSAL_DEBATE_ROUNDS", "3"))

        def _process_one(idx, item):
            # Build the text block for each result item
            print("\n-----------------causal analysis-----------------\n")
            print(f"paper {idx}/{len(paper_list)}: {item['title']}")
            # Use single-agent content-based analysis with bounded debate
            analysis = analyze_paper_content(item.get("content", ""), keywords=keywords, max_debate_rounds=DEBATE_ROUNDS)
            block = [
                f"【result: {idx}】",
                f"title: {item['title']}",
                # f"author: {item['authors']}",
                f"published: {item['published']}",
                # f"summary: {item['summary']}",
                f"url: {item['url']}",
                # f"{source}_id: {item[source+'_id']}",
                f"analysis: \n{analysis}",
                "---------------",
                ""
            ]
            structured = {
                "index": idx,
                "title": item.get("title"),
                "published": item.get("published"),
                "url": item.get("url"),
                "source": source,
                # "source_id": item.get(f"{source}_id"),
                "analysis": analysis
            }
            return idx, "\n\n".join(block), structured

        # Run analyses in parallel without changing per-paper logic
        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_process_one, idx, item) for idx, item in enumerate(paper_list, 1)]
            for fut in as_completed(futures):
                idx, block_text, structured = fut.result()
                results_map[idx] = (block_text, structured)

        # Reassemble results in original order
        for idx in range(1, len(paper_list) + 1):
            block_text, structured = results_map[idx]
            output_content.append(block_text)
            structured_results.append(structured)

        # with open("results.txt", "a+", encoding="utf-8") as f:
        #     f.write("\n".join(output_content))

        #print("Content saved to results.txt")
        return structured_results

def pre_analysis(paper,keywords):

    pre_inquiry_prompt = """You are a professional query specialist. Your task is to determine whether the specified scientific paper (provided via URL) explicitly mentions **all of the given keywords** or **semantically similar expressions**.
Requirements:
    Each keyword (or a clearly semantically equivalent expression) **must be explicitly extractable from the text**.
    No inference, vague matching, or speculative interpretation is allowed.
    **All** keywords or their semantic equivalents **must appear** in the paper. If any are missing, the answer is no.
Input:
    Keywords (one or more)
    Paper URL
Output Format:
    If the paper contains the keyword(s) or semantically similar expressions, output: yes
    Otherwise, output: no
Do not include any explanations, comments, or additional content of any kind.
"""
    pre_inquiry= f"""keywords: {keywords}. Paper:{paper["url"]}"""
    
    client = AgentClient()
    prompt = [{"role":"system","content":pre_inquiry_prompt},{"role":"user","content":pre_inquiry}]
    pre_inquiry_result = client.chat_completion(messages=prompt)
    return pre_inquiry_result

# Evidence and location validation for a single paper (strictly limited to Abstract/Discussion/Conclusion)
def validate_evidence(paper_url, analysis_text, keywords):
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

# Usage example
if __name__ == "__main__":
    #keywords = input("Please enter your search keywords: ").strip()
    user_input = input("Please enter your search keywords separated by semicolons ';': ").strip()
    keywords = [kw.strip() for kw in user_input.split(';') if kw.strip()]
    #start_date = input("Start date(YYYYMMDD, if not needed, please press Enter): ")
    #end_date = input("End date(YYYYMMDD, if not needed, please press Enter): ")
    papers = arxiv_search(keywords=keywords)
    agents_causal_analysis(paper_list=papers, keywords=keywords)
    print(papers)

