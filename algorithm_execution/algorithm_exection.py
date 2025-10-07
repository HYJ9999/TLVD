import importlib.util
import pandas as pd
import os
import sys

def load_algorithm_module(file_path):

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def exec_algorithm(algo_path, data_path):



    if not os.path.isfile(algo_path):
        print(f"Algorithm file not found: {algo_path}")
        return
    if not os.path.isfile(data_path):
        print(f"Data file not found: {data_path}")
        return


    try:
        data = pd.read_csv(data_path)
        print("Data loaded successfully")

    except Exception as e:
        print(f"Failed to load data: {e}")
        return


    try:
        algo_module = load_algorithm_module(algo_path)
        print(f"Algorithm module loaded successfully: {algo_path}")
    except Exception as e:
        print(f"Failed to load algorithm: {e}")
        return


    try:
        if hasattr(algo_module, 'run_algorithm'):
            result = algo_module.run_algorithm(data)
            print("Algorithm execution completed. Results:")
            print(result)
            return result
        else:
            print("Function 'run_algorithm(data)' not found in the algorithm module")
    except Exception as e:
        print(f"Error executing algorithm: {e}")

if __name__ == '__main__':
    exec_algorithm(algo_path="D:/python_projects/causal_inference/algorithm_execution/text.py",data_path="D:/python_projects/causal_inference/database_search/entity/dsp_vocab.csv")
