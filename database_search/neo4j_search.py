import re
from .agentClient import AgentClient
from .config import neo4j_config
from neo4j import GraphDatabase, exceptions

def neo4j_connect():
    driver = GraphDatabase.driver(neo4j_config.uri, auth=(neo4j_config.username, neo4j_config.password))
    return driver


def get_data(driver, query, parameters = None ):
    try:
        with driver.session(database=neo4j_config.database) as session:
            result = session.run(query, parameters)
            return [record for record in result]
    except exceptions.CypherSyntaxError as e:
        print(f"Cypher query error: {e.message}")
        return e.message
    except Exception as e:
        print(f"{str(e)}")
        return str(e)


def query_generate(usr_req=None, llm="deepseek-v3"):

    system_prompt = """你是一个neo4j数据库查询师，你的任务是: 1.根据用户的要求写出相应的neo4j数据库cypher查询代码。2.根据用户要求、查询代码以及错误信息，对错误查询代码进行修改。
    1.节点类型及其字段如下：
    Anatomy(id,name,uberon_id,bto_id,mesh_id,umls_cui,cl_id)
    Disease(id,name,do_id,kegg_id,pharmgkb_id,mesh_id,umls_cui,icd_10,icd_9,omim_id,iDISK_id)
    Drug(id,name,drugbank_id,kegg_id,pharmgkb_id,umls_cui,mesh_id,iDISK_id,CID)
    DSP(id,iDISK_id,name)
    Gene(id,symbol,hgnc_id,ncbi_id,pharmgkb_id,ensembl_id)
    Molecule(id,chembl_id,chebi_id,inchi,drugbank_id)
    Pathway(id,name,reactome_id,go_id,kegg_id)
    SDSI(id,iDISK_id,name)
    Side_Effect(id,umls_cui,name)
    Symptom(id,name,mesh_id,umls_cui,iDISK_id)
    TC(id,iDISK_id,name,umls_cui)
    其中的id是每个节点的主键，name、symbol或inchi是节点代表的实体的名称，其余的都是这一节点在其他数据库中的id标识。
    2.关系类型如下：
    ABSENT,ACTIVATES__STIMULATES,ADP_RIBOSYLATION_REACTION,AFFECTS_EXPRESSION_PRODUCTION_NEUTRAL,AGONISM__ACTIVATION,ALLEVIATES__REDUCES,ANTAGONISM__BLOCKING,ASSOCIATE,ASSOCIATION,BINDING__LIGAND_ESP.__RECEPTORS,BINDS,BIOMARKERS_DIAGNOSTIC,BIOMARKERS_OF_DISEASE_PROGRESSION,CARRIER,CAUSAL_MUTATIONS,CAUSE,CLEAVAGE_REACTION,COLOCALIZATION,COVARIES,DECREASES_EXPRESSION_PRODUCTION,DEPHOSPHORYLATION_REACTION,DIRECT_INTERATION,DOWNREGULATES,DRUG_TARGETS,EFFECT,ENHANCES_RESPONSE,ENZYME,ENZYME_ACTIVITY,EXPRESS,HAS_ADVERSE_EFFECT_ON,HAS_ADVERSE_REACTION,HAS_INGREDIENT,HAS_THERAPEUTIC_CLASS,IMPROPER_REGULATION_LINKED_TO_DISEASE,INCREASES_EXPRESSION_PRODUCTION,INFERRED_RELATION,INHIBITS,INHIBITS_CELL_GROWTH_ESP._CANCERS,INTERACTION,INTERACTS,INTERACTS_WITH,IS_A,IS_EFFECTIVE_FOR,METABOLISM__PHARMACOKINETICS,MUTATIONS_AFFECTING_DISEASE_COURSE,OVEREXPRESSION_IN_DISEASE,PALLIATES,PHOSPHORYLATION_REACTION,PHYSICAL_ASSOCIATION,POLYMORPHISMS_ALTER_RISK,POSSIBLE_THERAPEUTIC_EFFECT,PRESENT,PREVENTS__SUPPRESSES,PRODUCTION_BY_CELL_POPULATION,PROMOTES_PROGRESSION,PROTEIN_CLEAVAGE,REACTION,REGULATES,REGULATION,RESEMBLE,ROLE_IN_DISEASE_PATHOGENESIS,ROLE_IN_PATHOGENESIS,SAME_PROTEIN_OR_COMPLEX,SIGNALING_PATHWAY,TARGET,TRANSPORTER,TRANSPORT__CHANNELS,TREATMENT_THERAPY__INCLUDING_INVESTIGATORY,TREATS,UBIQUITINATION_REACTION,UPREGULATES
    每个关系类型的字段有Source、Inference_Score。其中Source是指该关系类型的来源(如CTD、KEGG)，Inference_Score是指两节点的关系类型与Source中其他数据库中的关系类型的相似程度。
    3.注意：
    -- 每个节点的这些id标识字段(如kegg_id、iDISK_id等)可能不存在，在写查询语句时“必须”注意校验。
    -- 关系类型中Inference_Score字段并可能不存在，在写查询语句时“必须”要注意校验。
    -- 查询返回的结果应保留用户关心的核心数据和查询涉及的标识属性信息(如id、name等)
    -- 查询代码必须以文本形式返回。
    -- **不要返回其他无关信息,例如解释性文字，只返回查询代码**。
    -- 特别注意关键词可能拥有多个名称，而数据库仅记录其中一种。此外，关键词的大小问题也需要注意，例如输入关键词为Alzheimer's disease，实际上在数据库中为alzheimer's disease。
    """
    if usr_req:
        client = AgentClient()
        response = client.execute(query=usr_req, system_prompt=system_prompt,llm=llm)

        print("----------query----------\n")
        print(response)
        print("--------------------")
        return response
    else:
        print("usr_req is empty")
        return None

def match_check(result: dict,llm="deepseek-v3"):
    system_prompt = """
    你是一个Neo4j查询结果评估器。你的任务是：
    1. 根据用户原始请求评估查询结果是否匹配
    2. 如果匹配，则只返回:*yes*。注意不要返回其他无关信息
    3. 如果不匹配，则返回:*no*,<原因>。
    """

    client = AgentClient()
    response = client.execute(query=str(result), system_prompt=system_prompt,llm=llm)

    return response

def database_query(usr_req, max_iter=5, llm="deepseek-v3"):
    history = [{"request":usr_req}]
    neo4j_driver = neo4j_connect()
    final_res = None
    for _ in range(max_iter):
        response = query_generate(usr_req=str(history), llm=llm).strip("```cypher").strip("```")
        result = get_data(driver=neo4j_driver, query=response)
        if isinstance(result, list):
            req_res = {"usr_req":usr_req, "query_result":result}
            is_match = match_check(req_res, llm=llm)
            
            pattern = r'^\*yes\*$|^\*no\*$'  # Only match "*yes*" or "*no*"
            if re.match(pattern, is_match):
                print(f"query result:\n{result}")
                final_res = result
                neo4j_driver.close()
                break
            else:
                history.append({"query":response, "response":result})
        else:
            history.append({"query":response, "response":result})

    return final_res        

# database_query(usr_req="Please help me write a Neo4j query to find the relationship between nodes A4GNT and VSTM5")