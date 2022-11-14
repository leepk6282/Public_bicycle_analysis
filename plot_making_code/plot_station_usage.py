import json
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath('.'))


def plot_borrow_return(config):
    """
    설명
    Args:
        config: configuration
    """
    preprocessing(config)
    return

def preprocessing(config):
    usage_path = config['path']['usage']
    file_list = os.listdir(usage_path)
    df = pd.read_csv(f'{usage_path}{file_list[0]}')
    # print(len(df), df.iloc[12000,:], sep='\n')

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    plot_borrow_return(config)
