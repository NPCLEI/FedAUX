
import ast
from copy import deepcopy
import math
import torch

from NPCLogger import NPCLog
from json import JSONDecoder
from torch.utils.data import _utils
from ControlPanel import ControlPanel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchDataset
from numpy.random import shuffle

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


from torchvision import transforms
class NPCDataset(torchDataset):

    @staticmethod
    def PICTransfer():
        return transforms.Compose([
        transforms.ToTensor(),transforms.Normalize([0.5], [0.5])
    ])

    def __init__(self,data_source = [],read_now = False) -> None:
        super(NPCDataset, self).__init__()
        self.dataset_item_label_idx = 1
        self.dataset_item_data_idx = 0
        self.dataset = []
        self._label_nums_ = {}
        self._label_idx_map_ = {}
        if data_source != None and len(data_source) != 0 and read_now:
            self.read(data_source)

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

    @staticmethod
    def tokenize(txt,max_length = 512,padding_chr = 0):
        tokenizer = NPCDataset.get_tokenizer()
        return NPCDataset.padding_tokens(tokenizer.encode(txt),max_length,padding_chr)

    @staticmethod
    def label2vector(idx,cls_num = ControlPanel.cur_ex_cls_num):
        res = torch.zeros(cls_num,dtype=torch.float32)
        res[idx] = 1
        return res

    @property
    def cls_num(self):
        return len(self.label_idx_map)

    @property
    def label_idx_map(self):
        """
            label -> label idx
        """
        if self._label_idx_map_ == {}:
            self.collect_label_idx_map()
        return self._label_idx_map_

    @property
    def data_static_by_label_nums(self):
        if self._label_nums_ == {}:
            self.collect_label_idx_map()
        return self._label_nums_

    @staticmethod
    def get_tokenizer():
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
        self.dataset = data_source
        return data_source

    def register_label_idx_map(self,label):
        if label not in self.label_idx_map:
            self.label_idx_map.setdefault(label,len(self.label_idx_map))
    
    def __clear_static__(self):
        self._label_idx_map_ = {}
        self._label_nums_ = {}

    def collect_label_idx_map(self):
        """
            statistics how many class of data to index
            self._label_idx_map_ = {class_name:index}
            self._label_nums_ = {class_name:nums}
        """
        self._label_idx_map_ = {}
        self._label_nums_ = {}
        for item in self.dataset:
            label = self.get_label(item)
            if label not in self._label_idx_map_:
                self._label_idx_map_.setdefault(label,len(self._label_idx_map_))
            if label not in self._label_nums_:
                self._label_nums_.setdefault(label,1)
            else:
                self._label_nums_[label] += 1

    def limit(self,limit_map = {}):
        if type(limit_map) == type(0):
            limit_map = dict([(key,limit_map) for key in self.label_idx_map.keys()])
        limit_table = dict([(key,0) for key in limit_map.keys()])
        new_dataset = []
        for idx in range(len(self)):
            item_label = self.get_label(self.dataset[idx])
            if limit_table[item_label] < limit_map[item_label]:
                limit_table[item_label] += 1
                new_dataset.append(self.dataset[idx])
        self.dataset = new_dataset
        self.__clear_static__()

    def get_sec_label(self,dataset_item):
        return dataset_item[2]

    def get_label(self,dataset_item):
        """
            dataset_item : self.dataset[idx] 
            ATTENTION: not self[idx]
        """
        return dataset_item[self.dataset_item_label_idx]

    def get_data(self,dataset_item):
        if type(dataset_item) == int and type(self.dataset[dataset_item]) != int:
            return self.get_data(self.dataset[dataset_item])
        return dataset_item[0]

    def get_shell(self):
        """
            object of data without the att of dataset
        """
        dataset = self.dataset
        self.dataset = []
        res = deepcopy(self)
        res.__clear_static__()
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

    @staticmethod
    def Merge(datas):
        shell = datas[0].get_shell()
        for data in datas:
            shell.dataset += data.dataset
        return shell

    @staticmethod
    def Split(data,test_ratio = 0.2,get_label = lambda item:item[1]):
        clsMap = {}
        for idx in range(len(data)):
            label = get_label(data[idx])
            if label not in clsMap:
                clsMap.setdefault(label,[])
            clsMap[label].append(data[idx])

        train,test = NPCDataset(),NPCDataset()
        for label in clsMap:
            mid = int(len(clsMap[label])*(1-test_ratio))
            train.dataset += clsMap[label][:mid]
            test.dataset += clsMap[label][mid:]
        return train,test

    def PrintDataInfo(self):
        clsmap = self.data_static_by_label_nums
        label2idx = self.label_idx_map
        print("%10s,%30s,%30s"%("label index","label name","label nums"))
        res = []
        for cls in label2idx:
            res.append((label2idx[cls],cls,clsmap[cls]))
        res.sort(key = lambda item:item[0])
        for item in res:
            print("%10s,%30s,%30s"%item)

    def shuffle(self):
        shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.get_data(item),self.get_label(item)





















class DoubleLoader:
    def __init__(self,data1:NPCDataset,data2:NPCDataset,batch_size = ControlPanel.batch_size,collate_fn = None):
        self.read_data1 = True
        self.data1 = data1
        self.cur_data1_idx = 0
        
        self.data2 = data2
        self.cur_data2_idx = 0

        self.batch_size = batch_size
        if collate_fn is None:
            collate_fn = _utils.collate.default_collate
  
        self.collate_fn = collate_fn

    def __len__(self):
        ld = len(self.data1) + len(self.data2)
        return math.ceil(ld/self.batch_size) 

    def __getitem__(self, idx):
        return self.get_next_batch()

    def fake_loader(self):
        return [i for i in range(len(self))]

    @property
    def data1readabel(self):
        return self.cur_data1_idx < len(self.data1)

    @property
    def data2readabel(self):
        return self.cur_data2_idx < len(self.data2)

    @property
    def readable(self):
        return self.data1readabel and self.data2readabel

    def reset(self):
        self.cur_data_idx = 0
        self.data1.shuffle()
        self.data2.shuffle()

    def __get_available_data__(self):
        if self.readable:
            self.reset()

        if self.read_data1 and self.data1readabel:
            choosed_data = self.data1
            cur_data_idx = self.cur_data1_idx
            self.cur_data1_idx += self.batch_size
        else:
            choosed_data = self.data2
            cur_data_idx = self.cur_data2_idx
            self.cur_data2_idx += self.batch_size

        self.read_data1 = not self.read_data1
        return choosed_data,cur_data_idx

    def get_next_batch(self):
        """
            调用该函数【后】,通过switch知道产生的是哪一类样本
        """
        choosed_data,cur_data_idx = self.__get_available_data__()

        lst,right = [],min(len(choosed_data),cur_data_idx + self.batch_size)
        for idx in range(cur_data_idx,right):
            lst.append((choosed_data[idx]))
            
        return self.collate_fn(lst)
    