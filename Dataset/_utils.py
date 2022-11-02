
import ast
from copy import deepcopy
import torch

from NPCLogger import NPCLog
from json import JSONDecoder
from ControlPanel import ControlPanel
from torch.utils.data import Dataset as torchDataset

def getJsonLikeDataParser(json_like_str):
    try:
        decoder = JSONDecoder()
        item = decoder.decode(json_like_str)
        return decoder.decode
    except:
        pass
    try:
        decoder = JSONDecoder()
        item = decoder.decode("[%s]"%json_like_str)
        return decoder.decode
    except:
        pass
    try:
        item = ast.literal_eval(json_like_str)
        return ast.literal_eval
    except:
        raise Exception("Can't find a suitable parser to decode the : '",json_like_str,"' to dict.")

def read_txt(file_path):
    try:
        txt = open(file_path,'r',encoding='utf8').read()
    except UnicodeDecodeError:
        try:
            txt = open(file_path,'r').read()
        except:
            return ""
    return txt

class NPCDataset(torchDataset):
    def __init__(self,data_source = None) -> None:
        super(NPCDataset, self).__init__()
        self.dataset = []
        self.label_idx_map = {}

    cls_id  = 101
    sep_id  = 102
    mask_id = 103
    
    @staticmethod
    def padding_tokens(tokens,max_length = 512,padding_chr = 0):
        len_tokens = len(tokens)
        if len_tokens < max_length:
            tokens += [padding_chr] * (max_length - len_tokens)
        tokens = tokens[:max_length]
        return tokens

    @property
    def cls_num(self):
        return len(self.label_idx_map)

    @property
    def tokenizer(self):
        return ControlPanel.Tokenizer()

    @staticmethod
    def read_txt_folder(folder_path,cls_name):
        import os
        txt_name_list = os.listdir(folder_path)
        dataset = []
        for name in txt_name_list:
            try:
                dataset.append((
                    "%s/%s"%(folder_path,name),
                    cls_name
                ))
            except:
                pass
        return dataset

    def read(self,data_source):
        return []

    def register_label_idx_map(self,label):
        if label not in self.label_idx_map:
            self.label_idx_map.setdefault(label,len(self.label_idx_map))

    def collect_label_idx_map(self):
        """
            statistics how many class of data to index
            self.label_idx_map = {class_name:index}
        """
        for item in self.dataset:
            label = self.get_label(item)
            if label not in self.label_idx_map:
                self.label_idx_map.setdefault(label,len(self.label_idx_map))

    def limit(self,limit_map = {}):
        limit_table = dict([(key,0) for key in limit_map.keys()])
        new_dataset = []
        for idx in range(len(self)):
            item_label = self.get_label(self.dataset[idx])
            if limit_table[item_label] < limit_map[item_label]:
                limit_table[item_label] += 1
                new_dataset.append(self.dataset[idx])
        self.dataset = new_dataset

    def get_sec_label(self,dataset_item):
        return dataset_item[2]

    def get_label(self,dataset_item):
        """
            dataset_item : self.dataset[idx] 
            ATTENTION: not self[idx]
        """
        return dataset_item[1]

    def get_data(self,dataset_item):
        if type(dataset_item) == int and type(self.dataset[dataset_item]) != int:
            return self.get_data(self.dataset[dataset_item])
        return dataset_item[0]

    def get_shell(self):
        """
            object of data without the att of dataset
        """
        dataset = self.dataset
        self.dataset = None
        res = deepcopy(self)
        res.dataset = []
        self.dataset = dataset
        return res

    def split(self,test_ratio = 0.2):
        clsMap = {}
        for idx in range(len(self)):
            item = self.dataset[idx]
            label = self.get_label(item)
            if label not in clsMap:
                clsMap.setdefault(label,[])
            clsMap[label].append(item)

        train,test = self.get_shell(),self.get_shell()
        for label in clsMap:
            mid = int(len(clsMap[label])*(1-test_ratio))
            train.dataset += clsMap[label][:mid]
            test.dataset += clsMap[label][mid:]
        return train,test


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]