import requests
import json
import pandas as pd
import numpy as np

def test(config):
    master_pd = make_master(config)
    preprocessing(config, master_pd)

def preprocessing(config, dataframe):
    """
    설명 (서울특별시 아닌거, 위도경도 00인거(폐쇄), 등등)
    Args:
        config:
        dataframe:
    """
    print(dataframe[dataframe.iloc[:,1].apply(lambda x : x.split(' ')[0])=='서울특별시'])
    

    print(dataframe)

def make_master(config):
    """
    설명
    Args:
        config: configuration
    References:
        bikeStatationList : 'http://data.seoul.go.kr/dataList/OA-21235/S/1/datasetView.do'
    """
    keys = config['keys']
    thousand = 0
    master_pd = pd.DataFrame()
    iters = True
    while iters:
        start_num, end_num = 1000*thousand+1, 1000*(thousand+1)
        response = requests.get(url=f"http://openapi.seoul.go.kr:8088/{keys}/json/bikeStationMaster/{start_num}/{end_num}/")
        if list(response.json().keys())[0]=='RESULT': # error
            print(response.json()['RESULT']['CODE'], response.json()['RESULT']['MESSAGE'], sep='\n')
            break
        else:
            for page in range(end_num-start_num+1):
                try:
                    master_json = response.json()['bikeStationMaster']
                    row_json = master_json[list(master_json.keys())[2]][page]
                    master_pd = pd.concat([master_pd, pd.DataFrame(row_json, index=[start_num+page])])
                except:
                    iters=False
                    break
        thousand+=1
    return master_pd

with open('config.json') as f:
    config = json.load(f)
test(config)
