import copy
from typing import List

import pandas as pd


class Score:

    def __init__(self, events, df=None) -> None:
        self.events = events
        if df is not None:
            self.load(df)

    def load(self, df: pd.DataFrame):
        for event in self.events:
            if event not in list(df.columns):
                raise Exception("Don't find event \"{0}\" in table".format(event))
        self.df = df

    def reload_events(self, reload_events, reload_df: pd.DataFrame):
        if len(self.df) != len(reload_df):
            raise Exception("Input table must have the same length with Score")

        for event in reload_events:
            if (event not in self.events) or (event not in reload_df.columns):
                raise Exception("Event \"{0}\" misses in table".format(event))
            self.df[event] = reload_df[event].values
            
    def average_events(self, reload_events, reload_df_list: list):
        assert type(reload_df_list) == list 

        for event in reload_events:
            for df in reload_df_list:
                if (event not in self.events) or (event not in df.columns):
                    raise Exception("Event \"{0}\" misses in table".format(event))
                self.df[event] = (df[event] + self.df[event]) 
            self.df[event] = self.df[event]/(1+len(reload_df_list))

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        else:
            return 0


class ScoreContainer:

    def __init__(self, events, score_buffer: dict = None) -> None:
        self.events = events
        self.score_dict = dict()    # dict[file, score]
        if score_buffer is not None:
            self.load(events, score_buffer)

    def load(self, events, score_buffer: dict):
        for file in score_buffer.keys():
            self.score_dict[file] = Score(events, score_buffer[file])
    
    def reload_events(self, reload_events, reload_container:'ScoreContainer'):
        for file, score in reload_container.score_dict.items():
            self.score_dict[file].reload_events(reload_events, score.df)
    
    def average_events(self, reload_events, reload_container_list: list)->'ScoreContainer':
        assert type(reload_container_list) == list
        res = copy.deepcopy(self)
        for file in res.files:
            score_df_list = [container.score_dict[file].df for container in reload_container_list]
            res.score_dict[file].average_events(reload_events, score_df_list)
        return res
    
        # for file, score in reload_container.score_dict.items():
        #     self.score_dict[file].average_events(reload_events, score.df)
    
    def get_score_buffer(self):
        buffer = {}
        for file, score in self.score_dict.items():
            buffer[file] = score.df
        return buffer
        
    @property
    def files(self):
        return list(self.score_dict.keys())
    
    def __len__(self):
        return len(self.score_dict)

def score_average(reload_events, reload_container_list:list)->'ScoreContainer':
    assert type(reload_container_list) == list
    if len(reload_container_list) == 1:
        return copy.deepcopy(reload_container_list[0])
    else:
        return reload_container_list[0].average_events(reload_events, reload_container_list[1:])