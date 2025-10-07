from collections import defaultdict, deque
import re
from typing import Dict, List, Tuple, Set

def variable_edge_extraction():

    with open("alpha0.05_rtscale1_N-1.dot", "r") as file:
        dot_content = file.read()


        edge_pattern = re.compile(r'(\w+)\s*->\s*(\w+)')
        edges = edge_pattern.findall(dot_content)


        is_hidden_variable = lambda x: re.match(r'^L\d+$', x)


        hidden_edges = [f"{src} -> {dst}" for src, dst in edges if is_hidden_variable(src) or is_hidden_variable(dst)]


        for edge in hidden_edges:
            print(edge)
        return hidden_edges
    

def edge_extraction(filepath):
    
    with open(filepath, "r", encoding="utf-8") as f:
        dot_content = f.read()
    edges = None

    directed_subgraph = re.search(r"subgraph Directed\s*{\s*edge \[.*?\]\s*(.*?)\s*}", dot_content, re.DOTALL)
    if directed_subgraph:
        edges_block = directed_subgraph.group(1)

        edges = re.findall(r'(\w+)\s*->\s*(\w+)', edges_block)


    else:
        print("No directed subgraph found.")
    
    return edges

def group_observed_by_latent(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Groups the list of edges by their latent node (source node), returning them in the format:
    [[ (L1, X1), (L1, X2) ], [ (L2, X3), (L2, X4) ], ...]
    """

    latent_groups = defaultdict(list)
    for src, tgt in edges:
        if re.fullmatch(r"L\d+", src):
            latent_groups[src].append(tgt)
    
    grouped_edges = [[(latent, obs) for obs in observed_list] for latent, observed_list in latent_groups.items()]
    return grouped_edges

def extract_latents_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Identify and return all edges involving latent variables (e.g., L1, L2, ...)
    by traversing the graph from bottom to top and grouping by latent variable
    while preserving hierarchical order (lower level appears later).
    """
    forward_graph = defaultdict(list)
    nodes = set()
    for src, tgt in edges:
        forward_graph[src].append(tgt)
        nodes.update([src, tgt])

    node_depth = {}

    def get_depth_from_src(node):
        if node in node_depth:
            return node_depth[node]
        if node not in forward_graph or not forward_graph[node]:
            node_depth[node] = 0
            return 0
        depth = max(get_depth_from_src(child) + 1 for child in forward_graph[node])
        node_depth[node] = depth
        return depth

    for node in nodes:
        get_depth_from_src(node)

    edge_index = {edge: i for i, edge in enumerate(edges)}
    latent_edges_with_level = []
    for src, tgt in edges:
        if re.fullmatch(r"L\d+", src):
            level = node_depth.get(src, 0)
            latent_edges_with_level.append((level, edge_index[(src, tgt)], src, tgt))

    # Step 1: Sort by level (from low to high), then by edge index to preserve original order
    latent_edges_with_level.sort()

    # Step 2: Group by latent variable within same level
    grouped = []
    level_grouped = defaultdict(list)
    for item in latent_edges_with_level:
        level_grouped[item[0]].append(item)

    for level in sorted(level_grouped):  # lower level comes later
        same_level_edges = level_grouped[level]

        # group by src (latent variable), preserve order
        by_latent = defaultdict(list)
        for entry in same_level_edges:
            _, _, src, tgt = entry
            by_latent[src].append((src, tgt))

        for latent in sorted(by_latent):  # optional: sort L1, L2, ...
            grouped.extend(by_latent[latent])
    
    return grouped

def replace_all_desc(edges: List[Tuple[str, str]], var_name_and_desc: dict, latent_dict:dict) -> List:
    """
    Replaces the node names within `edges` with their corresponding descriptions.

    Args:
        latent_dict (dict): A dictionary where the key is the node name and the value is the replacement name (used for latent variables).
        edges (List[Tuple[str, str]]): A list of edges, where each element is a tuple of (src, tgt).
        var_name_and_desc (dict): A dictionary where the key is the node name and the value is a dictionary containing 'name', 'description', and 'keywords'.

    Returns:
        List[str]: A list of the replaced edge descriptions, where each description is formatted as: "src_description;tgt_description".
    """

    output = []
    for src, tgt in edges:
        if src in latent_dict:
            src = latent_dict[src]
        if tgt in latent_dict:
            tgt = latent_dict[tgt]
        if src in  var_name_and_desc:
            src = var_name_and_desc[src].get("keywords", var_name_and_desc[src].get("description"))
        if tgt in  var_name_and_desc:
            tgt = var_name_and_desc[tgt].get("keywords", var_name_and_desc[tgt].get("description"))
        output.append(f"{src}||{tgt}")
    return output

def replace_observed_variables_desc(edges: List[Tuple[str, str]], var_name_and_desc: dict, latent_desc:dict = {}):
    """
Replaces the node names in `edges` with their corresponding descriptions.

Args:
    latent_dict (dict): A dictionary where the key is the node name and the value is the replacement name (used for latent variables).
    edges (List[Tuple[str, str]]): A list of edges, where each element is a tuple of (src, tgt).
    var_name_and_desc (dict): A dictionary where the key is the node name and the value is a dictionary containing 'name', 'description', and 'keywords'.

Returns:
    src: The latent variable.
    observed_variables_desc: A list of the replaced edge descriptions, where each description is formatted as a list of dictionaries, e.g., [{'tgt': tgt_description}, ...].
"""
    observed_variables_desc = []
    for src, tgt in edges:
        #print(src, tgt)
        if tgt in  var_name_and_desc:
            tgt_desc = var_name_and_desc[tgt].get("keywords", var_name_and_desc[tgt].get("description"))
        elif tgt in latent_desc:
            tgt_desc = latent_desc[tgt]
        else:
            tgt_desc = tgt
        observed_variables_desc.append({tgt:tgt_desc})    
    return src, observed_variables_desc

def save_papers_analysis_content(keywords,paper_list, filepath):
    save_content = []
    for idx, item in enumerate(paper_list, 1):

            block = [
                f"【result: {idx}】",
                f"title: {item['title']}",
                f"published: {item['published']}",
                f"url: {item['url']}",
                f"analysis: \n{item['analysis']}",
                "【end】",
                ""
            ]
            save_content.append("\n\n".join(block))
    with open(filepath, "a+", encoding="utf-8") as f:
        f.write("edge_validate: " + str(keywords) + "\n")
        f.write("\n".join(save_content))


def map_keywords_to_variable_keys(keywords: List[str], var_name_and_desc: Dict[str, dict], latents_desc: Dict[str, str]) -> List[str]:
    """
    Maps a keyword (which may be an observed variable's description/keywords value, or a latent variable's semantic description) to the variable's key.

    - **Observed Variables**: The 'description' or 'keywords' field maps to the variable key (name).
    - **Latent Variables**: The semantic description maps to the L-number.

    Includes a degree of **robustness**: converts to lowercase, compresses whitespace, removes leading/trailing quotes, and strips trailing punctuation (., ;, ,, ，, 。, ；, 、).
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().strip('"').strip("'").rstrip(".;,，。；、").lower()

    # Observed Variables: Normalized mapping of description/keywords to key
    rev_desc_to_key: Dict[str, str] = {}
    for _k, _meta in (var_name_and_desc or {}).items():
        _kw = _meta.get("keywords") if isinstance(_meta, dict) else None
        _ds = _meta.get("description") if isinstance(_meta, dict) else None
        if _kw:
            rev_desc_to_key[_norm(_kw)] = _k
        if _ds:
            rev_desc_to_key[_norm(_ds)] = _k

    # Latent Variable: Normalized mapping from a semantic description to an L-number
    latent_desc_to_key: Dict[str, str] = {}
    for _lk, _ldesc in (latents_desc or {}).items():
        if _ldesc is None:
            continue
        latent_desc_to_key[_norm(_ldesc)] = _lk

    def _to_key(term: str) -> str:
        n = _norm(term)
        if n in latent_desc_to_key:
            return latent_desc_to_key[n]
        if n in rev_desc_to_key:
            return rev_desc_to_key[n]
        return term

    return [_to_key(kw) for kw in (keywords or [])]


def group_edges_by_latent(edges: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Group edges by latent variable: {"L1":[(L1,X1),(L1,X2)],…}"""
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for src, tgt in edges:
        if re.fullmatch(r"L\d+", src):
            grouped[src].append((src, tgt))
        if re.fullmatch(r"X_\w+", src) and re.fullmatch(r"L\d+", tgt):
            grouped[tgt].append((src, tgt))
    return grouped


def select_latents_for_regeneration(all_latents: Dict[str, str], verified_ok: Set[str], failed_or_unverified: Set[str]) -> List[str]:
    """
        Select the list of latent variables whose descriptions need to be regenerated based on the current state.

        - Variables that are **successfully verified** (verified_ok) will **not** be regenerated.
        - Variables that have **failed or are unverified** (passed in as failed_or_unverified) **must** be regenerated.
    """

    current_latent_keys = set(all_latents.keys()) if all_latents else set()

    candidates = (failed_or_unverified | current_latent_keys) - verified_ok

    return sorted(candidates, key=lambda x: int(re.findall(r"\d+", x)[0]) if re.search(r"\d+", x) else 0)


def all_edges_of_latent_verified(latent: str, edges_for_latent: List[Tuple[str, str]], var_name_and_desc: Dict[str, dict], latents_desc: Dict[str, str], verify_edge_func) -> bool:
    """
    Determines if all outgoing edges (latent -> observed/latent) of a specific latent variable have been successfully verified.

    - **verify_edge_func**: Function signature verify_edge_func(keywords: List[str]) -> bool. This function is used to execute the verification for a single edge, returning whether it passed.
    - Uses the logic of **replace_all_desc** to substitute nodes with the keyword pairs used for retrieval/verification.
    """
    replaced_edges = replace_all_desc(edges_for_latent, var_name_and_desc, latents_desc)
    for edge in replaced_edges:
        keywords = [kw.rstrip('.') for kw in edge.split('||') if kw.rstrip('.')]
        ok = verify_edge_func(keywords)
        if not ok:
            return False
    return True


def update_verified_status(latent: str, success: bool, verified_ok: Set[str], failed_or_unverified: Set[str]):
    """Update the set of states based on the verification results of a single latent variable"""
    if success:
        verified_ok.add(latent)
        failed_or_unverified.discard(latent)
    else:
        failed_or_unverified.add(latent)
        verified_ok.discard(latent)
