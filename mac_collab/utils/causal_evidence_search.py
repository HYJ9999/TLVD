import json
from typing import List, Tuple, Dict, Any
import sys
import os
import re
from loguru import logger


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from paper_search import paper_search_analysis_single_agent, effectiveness_review
from Wikipedia_search import wiki_search
from utils.utils import replace_all_desc
from agentClient import AgentClient
from datetime import datetime
from utils.latent_desc_saver import load_and_update_latent_descriptions

def search_causal_evidence(
    edges: List[Tuple[str, str]], 
    var_names_and_desc: Dict[str, Any],
    latents_desc: Dict[str, str],
    agent_id: str,
    literature_source: str = "pmc",
    logger=None
) -> Tuple[Dict[Tuple[str, str], bool], float]:
    """
    Searches for causal evidence and evaluates the validity of each edge.

    Args:
        edges (List[Tuple[str, str]]): A list of edges, where each element is a (source, target) tuple.
        var_names_and_desc (dict): A dictionary of known variable descriptions.
        latents_desc (dict): A dictionary of latent variable descriptions.
        agent_id (str): The agent ID, used to differentiate saved files.
        literature_source (str): The literature source, either "pmc" or "arxiv".

    Returns:
        Tuple[Dict[Tuple[str, str], bool], float]:
            - A dictionary where the key is the edge (source, target) and the value is a boolean indicating whether evidence was found.
            - A float representing the ratio (proportion) of edges for which evidence was found.
    """
    client = AgentClient()
    edge_results = {}  
    edge_artifacts = {} 
    history=[]
    edge_valid_paper_count = {"valid":0,"total":0}
    edge_wiki_count = {"support":0, "total":0}

    latents_desc_new=load_and_update_latent_descriptions(latents_desc)
    
    
    verify_system_prompt = '''You are a causal relation verification expert.
Task:
You will receive:
A proposed causal relation in the format [X, Y], where X and Y are keywords.
A summary of causal analysis results from scientific papers.
A causal analysis result from a database.
An analysis of the retrieved Wikipedia content by an LLM.

Verification policy (lenient):
- You only need any one source (papers, database, or Wikipedia) to provide evidence(e.g.,paper analysis exists or wikipedia analysis returns yes), and you should answer "yes".
- If the database returns None or is irrelevant, disregard it and base the decision on the remaining sources.
- If none of the sources provide relevant evidence, answer "no".
- Ignore causal relationships that do not involve X and Y or their semantically equivalent expressions (e.g., synonyms or near-synonyms).

Input:
Keywords
Paper analysis
Database result
Wikipedia result

Output:
Only respond with "yes" or "no". Do not include any explanations, reasoning, or additional text.
'''
    
    
    for src, tgt in edges:
        
        replaced = replace_all_desc([(src, tgt)], var_names_and_desc, latents_desc_new)
        if not replaced:
            edge_results[(src, tgt)] = False
            continue
            
        keywords = [kw.rstrip('.') for kw in replaced[0].split('||') if kw.rstrip('.')]
        logger.info(f"Casual Search validate edge: {keywords}")
        

        literatures = []
        if literature_source == "arxiv":
            literatures = paper_search_analysis_single_agent.arxiv_search(keywords=keywords)
        elif literature_source == "pmc":
            literatures = paper_search_analysis_single_agent.pmc_search(keywords=keywords)
        else:
            print("literature_source error")
            edge_results[(src, tgt)] = False
            continue
            

        if not literatures:
            print(f"No literatures found for {keywords} from source={literature_source}")
            analysis_content = []
            aggregated_analysis_text = ""
            validate_info = [f"\nNo literature was retrieved"]
        else:
            analysis_content = paper_search_analysis_single_agent.agents_causal_analysis(
                paper_list=literatures, 
                keywords=keywords,
                source=literature_source
            )
            aggregated_analysis_text, validate_info, analysis_content,valid_count = effectiveness_review.paper_effectiveness_review(
                analysis_content, 
                keywords
            )
            # TODO: Record the number of valid papers for each edge and update accordingly
            
            logger.info(f"Agent {agent_id} validate edge: {src} -> {tgt} valid_count: {valid_count}")
            edge_valid_paper_count["valid"] =edge_valid_paper_count["valid"] + valid_count["valid"]
            edge_valid_paper_count["total"] =edge_valid_paper_count["total"] + valid_count["total"]
            
            save_edge_paper_ratio_record((src, tgt), valid_count, agent_id)
            update_edge_paper_ratio((src, tgt), valid_count, agent_id)



        wiki_result = wiki_search.find_causal_evidence(keywords[0], keywords[1])
    
        edge_wiki_count["total"] += 1
        pattern = r"^Yes,\s?.+"
        match = re.match(pattern, wiki_result)
        if match:
            edge_wiki_count["support"] += 1
        

        edge_artifacts[(src, tgt)] = {
            "keywords": keywords,
            "analysis_content": analysis_content,
            "validate_info": validate_info,
            "wiki_result": wiki_result
        }
        

        verify_query = f"""1.keywords
{keywords}
2.paper analysis
{aggregated_analysis_text}
3.database_result
{None}
4.wiki_analysis_result
{wiki_result}"""
        

        messages = [
            {"role": "system", "content": verify_system_prompt},
            {"role": "user", "content": verify_query}
        ]
        response = client.chat_completion(messages)
        if "yes" in response.strip().lower() :
            edge_results[(src, tgt)] = 'True'
        else: 
            edge_results[(src, tgt)] = 'False'

            history.append({"latents_desc": latents_desc, "edge": f"{src}->{tgt}", "error": verify_query})
        logger.info(f"Agent {agent_id} validate edge: {src} -> {tgt} result: {edge_results[(src, tgt)]}")
        

    with open(f'results_{agent_id}.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n=== Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for (src, tgt), artifacts in edge_artifacts.items():
            f.write(f"\nEdge: {src} -> {tgt}\n")
            f.write(f"Keywords: {artifacts['keywords']}\n")
            f.write("Analysis content:\n")
            for content in artifacts['analysis_content']:
                f.write(f"{content}\n")
            f.write(f"Validation info: {artifacts['validate_info']}\n")
            f.write(f"Wiki result: {artifacts['wiki_result']}\n")
            f.write("\n" + "="*50 + "\n")    


    success_rate = sum(1 for v in edge_results.values() if v) / len(edge_results) if edge_results else 0.0
    if edge_valid_paper_count["total"]!=0 and edge_wiki_count["total"]!=0:
        evidence_ratio=0.5*edge_valid_paper_count["valid"]/edge_valid_paper_count["total"] +0.5*edge_wiki_count["support"]/edge_wiki_count["total"]
    else:
        evidence_ratio= 0 +0.5*edge_wiki_count["support"]/edge_wiki_count["total"]
    
    return edge_results, success_rate,evidence_ratio


def save_edge_paper_ratio_record(
    edge: Tuple[str, str],
    valid_count: Dict[str, int],
    agent_id: str,
    file_path: str = "./mac_collab/data/edge_paper_ratio_record.json"
) -> None:
    """
    Appends and saves the ratio of valid papers for an edge to a JSON file.

    Args:
        edge (Tuple[str, str]): The edge tuple, formatted as (source, target).
        valid_count (dict): A dictionary containing 'valid' and 'total' counts, representing the number of valid papers and the total number of papers found.
        agent_id (str): The agent ID.
        file_path (str): The path to the JSON file where the data will be saved.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    

    edge_data = {
        "edge": f"{edge[0]}->{edge[1]}",
        "valid_papers": valid_count["valid"],
        "total_papers": valid_count["total"],
        "ratio": valid_count["valid"] / valid_count["total"] if valid_count["total"] > 0 else 0,
        "agent_id": agent_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:

        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
                logger.info(f"Created new record file: {file_path}")
                

        existing_data.append(edge_data)
        

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Appended data for edge {edge[0]}->{edge[1]}")
            
    except Exception as e:
        logger.error(f"Error saving valid paper ratio for edge: {str(e)}")

def update_edge_paper_ratio(
    edge: Tuple[str, str],
    valid_count: Dict[str, int],
    agent_id: str,
    file_path: str = "./mac_collab/data/edge_paper_ratio.json"
) -> None:
    """
    Overwrites and updates the ratio of valid papers for an edge in a JSON file.

    Args:
        edge (Tuple[str, str]): The edge tuple, formatted as (source, target).
        valid_count (dict): A dictionary containing 'valid' and 'total' counts, representing the number of valid papers and the total number of papers found.
        agent_id (str): The agent ID.
        file_path (str): The path to the JSON file where the data will be saved.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    edge_str = f"{edge[0]}->{edge[1]}"
    new_edge_data = {
        "edge": edge_str,
        "valid_papers": valid_count["valid"],
        "total_papers": valid_count["total"],
        "ratio": valid_count["valid"] / valid_count["total"] if valid_count["total"] > 0 else 0,
        "agent_id": agent_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:

        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
                logger.info(f"Created new statistics file: {file_path}")
        

        updated = False
        for i, data in enumerate(existing_data):
            if data["edge"] == edge_str and data["agent_id"] == agent_id:
                existing_data[i] = new_edge_data
                updated = True
                logger.info(f"Updated data for edge {edge_str}")
                break
        
        if not updated:
            existing_data.append(new_edge_data)
            logger.info(f"Added new data for edge {edge_str}")
        

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating valid paper ratio for edge: {str(e)}")