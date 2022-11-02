
import math
import torch

from statistics import mean
from Dataset.utils import NPCDataset,getJsonLikeDataParser

class AmazonReviewsJson(NPCDataset):

    def __init__(self, data_path = None , domain = None, read_now=False) -> None:
        super().__init__([], False)
        self.mode = 'SA'
        """
            |mode|func|
            |----|----|
            |SA|x,senti|
            |DC|x,domain|
            |SA&DC|x,senti,domain|
        """
        if read_now and data_path != None:
            self.read(data_path,domain)
        

    def read(self, data_source , domain , limit = 100000000):
        if type(data_source) != type("_"):
            self.dataset = self.dataset + data_source
            return True
        content = open(data_source,'r').readlines()
        # if len(content) < limit:
        #     return False
        parser = getJsonLikeDataParser(content[0])
        pad = 0
        for count,line in  enumerate(content):
            if count + pad + 1  > limit:
                break
            try:
                item = parser(line)
                input_ids = NPCDataset.tokenize(item["reviewText"])
                sentiment = 'pos' if int(item["overall"]) > 3 else 'neg'
                # sentiment = 1 if int(item["overall"]) > 3 else 0
                time = item["reviewTime"]
            except:
                pad -= 1
                continue
            item = [input_ids,sentiment,domain,time]
            self.dataset.append(item)
        return True

    def PrintDataInfo(self,title = True):
        domain = {}
        for item in self.dataset:
            _,s,d,__ = item
            if d not in domain:
                domain.setdefault(d,{'pos':0,'neg':0})
            if s in domain[d]:
                domain[d][s] += 1
            else:
                domain[d].setdefault(s,1)
        self.dataset_item_label_idx = -2
        self.collect_label_idx_map()
        label_idx_map = self.label_idx_map
        if title:
            print("%10s,%30s,%30s,%30s,%30s"%("label index","domain","pos","neg","label nums"))
        mean_lst = {
            'pos':[],'neg':[],'sum':[]
        }
        for d in domain:
            print("%10s,%30s,%30s,%30s,%30s"%(
                label_idx_map[d],d,domain[d]['pos'],domain[d]['neg'],self.data_static_by_label_nums[d]
            ))
            mean_lst['pos'].append(domain[d]['pos'])
            mean_lst['neg'].append(domain[d]['neg'])
            mean_lst['sum'].append(self.data_static_by_label_nums[d])
        print(mean_lst)
        print("%10s,%30s,%30.2lf,%30.2lf,%30.2lf"%(
                'mean','',mean(mean_lst['pos']),mean(mean_lst['neg']),mean(mean_lst['sum'])
            ))
        self.dataset_item_label_idx = 1
        self.__clear_static__()
        
    def get_label(self, dataset_item):
        if self.dataset_item_label_idx == 1:
            senti = 1 if dataset_item[self.dataset_item_label_idx] == 'pos' else 0
            # senti = torch.tensor(senti,dtype=torch.int64)
            return senti
        else:
            return dataset_item[self.dataset_item_label_idx]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx][self.dataset_item_data_idx],dtype=torch.int64)
        senti = 1 if self.dataset[idx][self.dataset_item_label_idx] == 'pos' else 0
        senti = torch.tensor(senti,dtype=torch.int64)
        return input_ids,senti