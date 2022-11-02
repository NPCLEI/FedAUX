
from NPCLogger import NPCLog
import torch
from torch.utils.data import Dataset as torchDataset
import ControlPanel
if ControlPanel.tokenizer == None:
    ControlPanel.ReadBERT(b = False)
cls_id = ControlPanel.tokenizer.vocab["[CLS]"]
sep_id = ControlPanel.tokenizer.vocab["[SEP]"]
msk_id = ControlPanel.tokenizer.vocab["[MASK]"]

import json
class JsonData(torchDataset):
    def __init__(self,file_path,distillModel = False):
        super(JsonData, self).__init__()
        self.limitedLen = 512
        self.dataset = []
        if file_path == None:            
            return
        NPCLog("read data...",end = '')
        if not distillModel:
            self.dataset = json.load(open(file_path,"r"))
        else:
            self.readJosnLike(file_path)
        self.idxes = []
        self.distillModel = distillModel
        NPCLog("done.",title="")
    
    def readJosnLike(self,path):
        import ast
        content = open(path,'r').readlines()
        for line in content:
            item = ast.literal_eval(line)
            self.dataset.append(item)

    def balance(self):
        label_neg = []
        label_pos = []
        for item in self.dataset:
            if int(item["overall"]) > 3:
                label_pos.append(item)
            else:
                label_neg.append(item)
        #去个位样本，防止Cuda出现问题？具体怎样引发的问题未知。
        nll = len(label_neg)
        label_neg = label_neg[:nll - int(str(nll)[-1])]
        ##############################################
        pll,nll = len(label_pos),len(label_neg)
        if nll < pll:
            label_pos = label_pos[:nll]
            self.dataset = label_pos + label_neg
        NPCLog("balance result: (pos,%d) (neg,%d)"%(len(label_pos),len(label_neg)))

    def __len__(self):
        return len(self.dataset)

    def split(self,test_ratio = 0.2):
        mid_ = 1 - test_ratio
        label_True = []
        label_Fals = []
        def cutList(lst):
            mid = int(len(lst) * mid_)
            train,test = lst[:mid] , lst[mid:]
            print(len(lst) , mid,len(train),len(test))
            return train,test
            
        for item in self.dataset:
            if int(item["overall"]) > 3:
                label_True.append(item)
            else:
                label_Fals.append(item)

        ltt,lte = cutList(label_True)
        lft,lfe = cutList(label_Fals)

        train,test = JsonData(None),JsonData(None)
        train.dataset = ltt + lft
        test.dataset = lte + lfe
        return train,test

    def collate_func(self,batch_dic):
        if self.distillModel:
            xs,ys,ps = [],[],[]
            for x,y,p in batch_dic:
                if type(x) == type(-1):
                    continue
                xs.append(x.unsqueeze(dim=0))
                ys.append(y)
                ps.append(p)
            return torch.cat(xs,dim=0),torch.cat(ys,dim=0),torch.cat(ps,dim=0)
        xs,ys = [],[]
        for x,y in batch_dic:
            if type(x) == type(-1):
                continue
            xs.append(x.unsqueeze(dim=0))
            ys.append(y)
        return torch.cat(xs,dim=0),torch.cat(ys,dim=0)

    @staticmethod
    def paraData(dataitem,limitedLen = 512):
        try:
            label,txt = int(dataitem["overall"]) > 3 , dataitem["reviewText"]
            tl = 1 if label else 0
            # inputs = Config.tokenizer(txt ,padding = 'max_length', return_tensors='pt',truncation  = True).to(Config.device)
            # inputs = Config.tokenizer.encode_plus(txt ,padding = 'max_length', return_tensors='pt',truncation  = True)
            tokens = ControlPanel.tokenizer.encode(txt)
            res = [cls_id] + tokens + [sep_id]
            ls = len(res)
            if ls < limitedLen:
                # NPCLog(CsvDataset.count)
                for _ in range(limitedLen - ls):
                    res.append(0)
            res = res[:limitedLen]
            # NPCLog(inputs)
            # with torch.no_grad():
            #     embedding = Config.bert(**inputs).last_hidden_state
            return torch.tensor(res,dtype=torch.int64).to(ControlPanel.device),torch.Tensor([[tl,1-tl]]).to(ControlPanel.device)
            
        except Exception as e:
            # NPCLog("[npc report] Unhandle Eorr:",e,"auto handle:","skip")
            return -1,-1

    def __getitem_with_distill__(self, idx: int):
        try:
            item = self.dataset[idx]
            _,txt = int(item["overall"]) > 3 , item["reviewText"]
            gp = [float(v) for v in item["gobal_predict"]]
            prediect_label = [
                1 if gp[0] > gp[1] else 0,
                1 if gp[1] > gp[0] else 0
            ]
            tokens = ControlPanel.tokenizer.encode(txt)
            res = [cls_id] + tokens + [sep_id]
            ls = len(res)
            if ls < self.limitedLen:
                # NPCLog(CsvDataset.count)
                for _ in range(self.limitedLen - ls):
                    res.append(0)
            res = res[:self.limitedLen]
            return torch.tensor(res,dtype=torch.int64).to(ControlPanel.device),torch.Tensor([prediect_label]).to(ControlPanel.device),torch.Tensor([gp]).to(ControlPanel.device)
            
        except Exception as e:
            # NPCLog("[npc report] Unhandle Eorr:",e,"auto handle:","skip")
            return -1,-1


    def __getitem__(self, idx: int):
        if self.distillModel:
            return self.__getitem_with_distill__(idx)
        try:
            label,txt = int(self.dataset[idx]["overall"]) > 3 , self.dataset[idx]["reviewText"]
            tl = 1 if label else 0
            # inputs = Config.tokenizer(txt ,padding = 'max_length', return_tensors='pt',truncation  = True).to(Config.device)
            # inputs = Config.tokenizer.encode_plus(txt ,padding = 'max_length', return_tensors='pt',truncation  = True)
            tokens = ControlPanel.tokenizer.encode(txt)
            res = [cls_id] + tokens + [sep_id]
            ls = len(res)
            if ls < self.limitedLen:
                # NPCLog(CsvDataset.count)
                for _ in range(self.limitedLen - ls):
                    res.append(0)
            res = res[:self.limitedLen]
            # NPCLog(inputs)
            # with torch.no_grad():
            #     embedding = Config.bert(**inputs).last_hidden_state
            return torch.tensor(res,dtype=torch.int64).to(ControlPanel.device),torch.Tensor([[tl,1-tl]]).to(ControlPanel.device)
            
        except Exception as e:
            # NPCLog("[npc report] Unhandle Eorr:",e,"auto handle:","skip")
            return -1,-1


""""
                            All_Beauty_5.test.json       784      17

                           All_Beauty_5.train.json      3134      66

                        AMAZON_FASHION_5.test.json       159      34

                       AMAZON_FASHION_5.train.json       633     135

                            Appliances_5.test.json       282      89

                           Appliances_5.train.json      1125     352

                Arts_Crafts_and_Sewing_5.test.json      65281   8146

               Arts_Crafts_and_Sewing_5.train.json      261122  32583

                            Automotive_5.test.json      77382   8863

                           Automotive_5.train.json      309524  35448

                         CDs_and_Vinyl_5.test.json      17058    753

                        CDs_and_Vinyl_5.train.json      68232   3010

           Cell_Phones_and_Accessories_5.test.json      22763   4499

          Cell_Phones_and_Accessories_5.train.json      91052   17995

            Clothing_Shoes_and_Jewelry_5.test.json      1303348 313378

           Clothing_Shoes_and_Jewelry_5.train.json      5213388 1253509

                         Digital_Music_5.test.json      28600   1980

                        Digital_Music_5.train.json      114396  7919

                           Electronics_5.test.json      882192  207657

                          Electronics_5.train.json      3528768 830628

                            Gift_Cards_5.test.json       303       7

                           Gift_Cards_5.train.json      1211      25

              Grocery_and_Gourmet_Food_5.test.json      43217   5396

             Grocery_and_Gourmet_Food_5.train.json      172864  21581

                      Home_and_Kitchen_5.test.json      810797  166988

                     Home_and_Kitchen_5.train.json      3243184 667948

             Industrial_and_Scientific_5.test.json      9808    1226

            Industrial_and_Scientific_5.train.json      39229   4903

                          Kindle_Store_5.test.json      337452  54927

                         Kindle_Store_5.train.json      1349808 219705

                         Luxury_Beauty_5.test.json      4175     928

                        Luxury_Beauty_5.train.json      16698   3711

                Magazine_Subscriptions_5.test.json       320      76

               Magazine_Subscriptions_5.train.json      1279     303

                         Movies_and_TV_5.test.json      479198  130366

                        Movies_and_TV_5.train.json      1916791 521462

                   Musical_Instruments_5.test.json      30931   4687

                  Musical_Instruments_5.train.json      123723  18747

                       Office_Products_5.test.json      104010  16194

                      Office_Products_5.train.json      416038  64775

                 Patio_Lawn_and_Garden_5.test.json      30867   4930

                Patio_Lawn_and_Garden_5.train.json      123468  19717

                          Pet_Supplies_5.test.json      78582   14564

                         Pet_Supplies_5.train.json      314326  58254

                          Prime_Pantry_5.test.json      14780   1918

                         Prime_Pantry_5.train.json      59117   7668

                              Software_5.test.json      1702     732

                             Software_5.train.json      6807    2928

                   Sports_and_Outdoors_5.test.json      360030  62773

                  Sports_and_Outdoors_5.train.json      1440116 251090

            Tools_and_Home_Improvement_5.test.json      83265   10059

           Tools_and_Home_Improvement_5.train.json      333060  40235

                        Toys_and_Games_5.test.json      58870   5725

                       Toys_and_Games_5.train.json      235476  22898

                           Video_Games_5.test.json      6940    1020

                          Video_Games_5.train.json      27758   4078

"""