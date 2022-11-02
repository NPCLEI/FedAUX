import sys
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../..")

import torch.nn as nn
from ControlPanel import bert_config
from pytorch_transformers import BertPreTrainedModel,BertModel
from Models.nn import NPCModule
from Models.SABert.AmazonReviews import JsonData

# 定义网络结构
class SABert(BertPreTrainedModel,NPCModule):
    def __init__(self,name = '',config = bert_config):
        BertPreTrainedModel.__init__(self,config)
        NPCModule.__init__(self,name = name)
        self.bert = BertModel(config)
        self.mlpInLen = 768
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.mlpInLen),
            nn.Linear(self.mlpInLen,1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024,2),
        )

    def get_parameters(net,bertLR = 1e-6,LR = 1e-4):
        return [
                {'params': net.bert.parameters(), 'lr': bertLR},
                {'params': net.mlp.parameters(), 'lr': LR}
            ]

    def forward(self, tokens , outHid = False):
        x,p = self.bert(tokens)
        if outHid:
            return x,self.mlp(p)
        else:
            return torch.softmax(self.mlp(p),dim = 1)

    @staticmethod
    def cmp(o,p):
        # print(torch.argmax(o,dim=1),torch.argmax(p,dim=1))
        return torch.argmax(o,dim=1) == torch.argmax(p,dim=1)

if __name__ == "__main__":
    import utils
    # datasetnames = ['All_Beauty_5.json', 'AMAZON_FASHION_5.json', 'Appliances_5.json', 'Arts_Crafts_and_Sewing_5.json', 'Automotive_5.json', 'CDs_and_Vinyl_5.json', 'Cell_Phones_and_Accessories_5.json', 'Clothing_Shoes_and_Jewelry_5.json', 'Digital_Music_5.json', 'Electronics_5.json', 'Gift_Cards_5.json', 'Grocery_and_Gourmet_Food_5.json', 'Home_and_Kitchen_5.json', 'Industrial_and_Scientific_5.json', 'Kindle_Store_5.json', 'Luxury_Beauty_5.json', 'Magazine_Subscriptions_5.json', 'Movies_and_TV_5.json', 'Musical_Instruments_5.json', 'Office_Products_5.json', 'Patio_Lawn_and_Garden_5.json', 'Pet_Supplies_5.json', 'Prime_Pantry_5.json', 'Software_5.json', 'Sports_and_Outdoors_5.json', 'Tools_and_Home_Improvement_5.json', 'Toys_and_Games_5.json', 'Video_Games_5.json']
    datasetnames = ['Grocery_and_Gourmet_Food_5','Industrial_and_Scientific_5','Software_5','Luxury_Beauty_5']
    pathsp = r'E:\Dataset\NLP-TC-DOC-Level\AmazonReview_V2018\SA2018and2017SP'
    
    def initServerModel():
        sabert = SABert.from_pretrained("bert-base-uncased")
        sabert.setFunc(torch.nn.BCELoss(),SABert.cmp,SABert.cmp)
        return sabert

    # or_sabert = utils.CheckModel("or_sabert",initServerModel)

    import os
    # client_models = []
    for datasetname in datasetnames:
        if os.path.exists("%s/%s.train.json"%(pathsp,datasetname[:-5])):
            continue
        train,test = JsonData("%s/%s.train.json"%(pathsp,datasetname)),JsonData("%s/%s.test.json"%(pathsp,datasetname))
        train.balance()
        test.balance()
        or_sabert = utils.CheckModel("or_sabert",initServerModel)
        net.name = datasetname
        net = utils.CheckModel(datasetname,lambda:or_sabert.Train(train,test),saveModel=False)
        # client_models.append(net)
    
