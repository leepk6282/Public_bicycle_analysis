import requests
import json
import pandas as pd
import time
import os

def bikeStationList(config):
    """
    따릉이 대여소 DB 생성 및 전처리
    Args:
        config: configuration
    """
    master_pd = make_df(config)
    preprocessed_station, not_in_Seoul, closed_station = preprocessing(master_pd)
    return preprocessed_station, not_in_Seoul, closed_station

def preprocessing(raw_df):
    """
    따릉이 대여소 전처리 과정
    Args:
        config:
        raw_df:
    Results:
        preprocessed_station, not_in_Seoul, closed_station
    """
    # only include station in Seoul
    _Si_Do = raw_df.iloc[:,0].apply(lambda x: str(x).split(' ')[0])
    not_in_Seoul = raw_df[_Si_Do == '경기']

    # only include open station
    _LNT, _addr = raw_df.iloc[:,-1], raw_df.iloc[:,0]
    closed_station = raw_df[(_LNT==0) | (_addr=='') | _addr.isna()]

    # classify 'Si' to 'Gu'
    preprocessed_station = raw_df[(_Si_Do!='경기') & (_LNT!=0) & (_addr!='') & ~(_addr.isna())]
    preprocessed_station.insert(len(preprocessed_station.columns), 'Gu',
                                    preprocessed_station.iloc[:,0].apply(lambda x: str(x).split(' ')[1]))
    return preprocessed_station, not_in_Seoul, closed_station

def make_df(config):
    """
    따릉이 대여소 전체 리스트 생성
    Args:
        config: configuration
    References:
        bikeStatationList : 'http://data.seoul.go.kr/dataList/OA-21235/S/1/datasetView.do'
    """
    keys = config['keys']
    data_path = config['path']['data']
    os.makedirs(data_path, exist_ok=True)
    if 'bikeStationList.csv' in os.listdir(data_path):
        master_pd = pd.read_csv(f'{data_path}bikeStationList.csv', index_col=[0])
    else:
        start_time = time.time()
        thousand = 0
        master_pd = pd.DataFrame()
        iters = True
        while iters:
            start_idx, end_idx = 1000*thousand+1, 1000*(thousand+1)
            response = requests.get(url=f"http://openapi.seoul.go.kr:8088/{keys}/json/bikeStationMaster/{start_idx}/{end_idx}/")
            if list(response.json().keys())[0]=='RESULT': # error
                print(response.json()['RESULT']['CODE'], response.json()['RESULT']['MESSAGE'], sep='\n')
                break
            else:
                for page in range(end_idx-start_idx+1):
                    try:
                        master_json = response.json()['bikeStationMaster']
                        row_json = master_json[list(master_json.keys())[2]][page]
                        master_pd = pd.concat([master_pd, pd.DataFrame(row_json, index=[start_idx+page])])
                    except:
                        iters=False
                        break
            thousand+=1
        master_pd = master_pd.set_index(master_pd.columns[0])
        master_pd.to_csv(f'{data_path}bikeStationList.csv', index=True)
        print(f'대여소 목록 생성 {time.time()-start_time:<.2f}초 소요')
    return master_pd

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    bikeStationList(config)
