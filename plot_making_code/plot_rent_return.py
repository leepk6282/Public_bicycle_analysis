import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath('.'))
from data_making_code.bikeStationList import bikeStationList
import matplotlib.pyplot as plt

def plot_borrow_return(config):
    """
    설명
    Args:
        config: configuration
    """
    result_df = preprocessing(config)
    make_plot(config, result_df)
    return

def make_plot(config, array:pd.DataFrame):
    """
    설명
    Args:
        config:
        array: pd.DataFrame.shape = (*,3), columns = ['rent','return','rent-return']
    """
    save_path = config['path']['plot']
    fig, ax = plt.subplots(figsize = (12,6))
    ax.set_title('Counts')
    ax.set_xlabel('Stations')
    ax.set_ylabel('Counts')
    for idx, mode in enumerate(['Rent', 'Return']):
        array = array.sort_values(by=[array.columns[idx]], ascending=False)
        ax.plot(range(len(array)), array.iloc[:,idx], '-', label=f'{mode} Counts', color=config['colors'][idx*3+3])
    desc = array.iloc[:,idx].describe()
    ax.axhline(y=desc[1]+desc[2]*3, linestyle='--', color='red',label='mean+3*std')
    ax.legend(loc='upper right')
    plt.savefig(f"{save_path}/demand_sort.png")

    fig, ax = plt.subplots(figsize = (12,6))
    ax.set_title('Imbalance of demand and supply')
    ax.set_xlabel('Stations')
    ax.set_ylabel('Rents - Returns')
    array = array.sort_values(by=[array.columns[2]], ascending=False)
    ax.plot(range(len(array)), array.iloc[:,2], '-', label='Imbalance', color=config['colors'][6])
    ax.legend(loc='upper right')
    plt.savefig(f"{save_path}/imbalance_sort.png")

def pivot_and_sum(config, indices):
    """
    pivot_table 만들고 sum 진행 결과 반환
    Args:
        config:
        indices
    """
    usage_path = config['path']['usage']
    file_list = os.listdir(usage_path)
    result_df_rent = pd.DataFrame()
    result_df_return = pd.DataFrame()
    for file in tqdm(file_list):
        file_df = pd.read_csv(f'{usage_path}{file}')
        file_pivot_rent = pd.pivot_table(file_df, index= 'RENT_STATION_ID', values='SEX_CD', aggfunc='count')
        file_pivot_return = pd.pivot_table(file_df, index= 'RETURN_STATION_ID', values='SEX_CD', aggfunc='count')
        result_df_rent = pd.concat([result_df_rent, file_pivot_rent], axis=1)
        result_df_return = pd.concat([result_df_return, file_pivot_return], axis=1)
    result_df_rent = pd.DataFrame(result_df_rent.sum(axis=1), index= result_df_rent.index, columns=['rent'])
    result_df_return = pd.DataFrame(result_df_return.sum(axis=1), index= result_df_return.index, columns=['return'])
    running_sr_rent = pd.Series(result_df_rent.index, index=result_df_rent.index).apply(lambda x: x in indices)
    running_sr_return = pd.Series(result_df_return.index, index=result_df_return.index).apply(lambda x: x in indices)
    result_df_rent = result_df_rent.loc[running_sr_rent]
    result_df_return = result_df_return.loc[running_sr_return]
    result_df = pd.concat([result_df_rent, result_df_return], axis=1)
    return result_df

def preprocessing(config):
    preprocessed_station, _, _ = bikeStationList(config)
    result_df = pivot_and_sum(config, indices=preprocessed_station.index)
    # result_df['rent-return'] = result_df.iloc[:,0] - result_df.iloc[:,1]
    result_df.insert(2, 'rent-return', result_df.iloc[:,0] - result_df.iloc[:,1], True)
    return result_df

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    plot_borrow_return(config)