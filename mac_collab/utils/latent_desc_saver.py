import json
import os
from typing import Dict
from datetime import datetime
from loguru import logger

def save_latent_description(
    latents_desc: Dict[str, str], 
    file_path: str = "./mac_collab/data/hidden_var_descriptions.json",
    mode: str = "overwrite"  # 'append' or 'overwrite'
) -> None:
    """
    Saves latent variables and their descriptions to a JSON file.

    Args:
        latents_desc (dict): A dictionary of latent variables and their descriptions, formatted as {"L1": "description", ...}.
        file_path (str): The path to the file where data will be saved. Defaults to "./mac_collab/data/hidden_var_descriptions.json".
        mode (str): The saving mode. Available options are:
            - "append": Retains the original description for existing keys (default mode).
            - "overwrite": Overwrites the original description with the new description for existing keys.
    """
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        
        existing_desc = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    existing_desc = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Unable to parse existing file {file_path}, creating a new file")
        else:
            
            logger.info(f"File {file_path} does not exist, creating a new file")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

       
        for key, new_value in latents_desc.items():
            if key in existing_desc:
                old_value = existing_desc[key]
                if old_value != new_value:
                    if mode == "overwrite":
                        logger.info(f"Overwriting description for latent variable {key}:")
                        logger.info(f"  - Old description: {old_value}")
                        logger.info(f"  - New description: {new_value}")
                        existing_desc[key] = new_value
                    else:  # append 
                        logger.info(f"Keeping original description for latent variable {key}: {old_value}")
                        logger.info(f"Ignoring new description: {new_value}")
            else:
                
                existing_desc[key] = new_value
                logger.info(f"Added new latent variable {key}: {new_value}")

        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_desc, f, ensure_ascii=False, indent=2)
        
       
        logger.info(f"Save mode: {'overwrite' if mode == 'overwrite' else 'append'}")
        logger.info(f"Latent variable descriptions saved to {file_path}")
        logger.info(f"Total {len(existing_desc)} latent variable descriptions")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error occurred while saving latent variable descriptions: {str(e)}")
        raise

def append_latent_descriptions_to_txt(
    descriptions: Dict[str, str],
    agent_id: str = "agent_1"
) -> None:
    """
    Appends the latent variable dictionary and its descriptions to a TXT file.

    Args:
        descriptions (dict): A dictionary of latent variables and their descriptions, formatted as {"L1": "description", ...}.
        txt_file (str): The path to the TXT file where data will be saved. Defaults to "./mac_collab/data/latent_descriptions.txt".
    """
    try:
       
        os.makedirs(os.path.dirname(f"./mac_collab/data/results_{agent_id}.txt"), exist_ok=True)
        
        
        if not os.path.exists(f"./mac_collab/data/results_{agent_id}.txt"):
            with open(f"./mac_collab/data/results_{agent_id}.txt", "w", encoding="utf-8") as f:
                f.write("# Latent variable description records\n")
                f.write("# Created at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("=" * 50 + "\n\n")
        
        
        with open(f"./mac_collab/data/results_{agent_id}.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== Latent variable descriptions {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            for var_name, description in descriptions.items():
                f.write(f"{var_name}: {description}\n")
            f.write("="*50 + "\n")
        
        logger.info(f"Appended {descriptions} latent variable descriptions to ./mac_collab/data/results_{agent_id}.txt")
        
    except Exception as e:
        logger.error(f"Error occurred while appending latent variable descriptions: {str(e)}")


def load_and_update_latent_descriptions(
    new_descriptions: Dict[str, str],
    file_path: str = "./mac_collab/data/hidden_var_descriptions.json"
) -> Dict[str, str]:
    """
    Loads the latent variable description file, updates any matching keys with the new descriptions, and returns the updated dictionary (without saving it to the file).

    Args:
        new_descriptions (dict): A dictionary of new latent variable descriptions, formatted as {"L1": "new description", ...}.
        file_path (str): The path to the description file. Defaults to "./mac_collab/data/hidden_var_descriptions.json".
        
    Returns:
        Dict[str, str]: The updated dictionary containing all latent variables and their descriptions.
    """
    try:
        
        existing_desc = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    existing_desc = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Unable to parse file {file_path}, using empty dictionary")
        else:
            
            logger.info(f"File {file_path} does not exist, creating a new file")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        
        updated_desc = existing_desc.copy()
        updated_count = 0
        for key, new_value in new_descriptions.items():
            if key in updated_desc:
                old_value = updated_desc[key]
                if old_value != new_value:
                    updated_desc[key] = new_value
                    updated_count += 1
                    logger.info(f"Updated description for latent variable {key}:")
                    logger.info(f"  - Old description: {old_value}")
                    logger.info(f"  - New description: {new_value}")
            else:
                
                updated_desc[key] = new_value
                logger.info(f"Added latent variable {key}: {new_value}")
        
        logger.info(f"Returning existing latent variable descriptions for causal evidence search")
        return updated_desc
                
    except Exception as e:
        logger.error(f"Error occurred while updating latent variable descriptions: {str(e)}")
        return {}


# Save analysis content
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

# Compute success rate