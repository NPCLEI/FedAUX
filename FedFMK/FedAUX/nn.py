
import torch

from torch import nn
from Models.nn import NPCModule
from pytorch_transformers import BertModel

class ClassifyShell(NPCModule):
    def __init__(self,feature_extractor:NPCModule,cls_num,feature_dim = 128,name=''):
        NPCModule.__init__(self,name)
        self.classifyer = nn.Sequential(
            nn.Linear(feature_dim,1024),nn.ReLU(),nn.LayerNorm(1024),
            nn.Linear(1024,1024),nn.ReLU(),nn.LayerNorm(1024),
            nn.Linear(1024,cls_num),nn.ReLU(),
        )
        self.feature_extractor = feature_extractor
        # self.feature_extractor = BertModel()
        self.feature_extractor.encoder.output_hidden_states = False
        self.train_feature_extractor = False
        self.lr = 1e-5
        self.distill_mode = False
        self.output_distribution = False

    def get_parameters(net,bert_lr = 1e-7):
        if net.train_feature_extractor or net.distill_mode:
            return [
                    {'params': net.feature_extractor.encoder.parameters(), 'lr': bert_lr},
                    {'params': net.feature_extractor.pooler.parameters(), 'lr': bert_lr},
                    {'params': net.classifyer.parameters(), 'lr': net.lr}
                ]
        else:
            return [{'params': net.classifyer.parameters(), 'lr': net.lr},]

    def forward(self,input_ids):
        _,feature = self.feature_extractor(input_ids)
        if self.distill_mode:
            return self.classifyer(feature).log_softmax(1)
        if self.output_distribution:
            return torch.softmax(self.classifyer(feature),1)
        return self.classifyer(feature)