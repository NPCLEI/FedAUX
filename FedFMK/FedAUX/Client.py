
from statistics import mean
import torch
import utils

from copy import deepcopy
from ControlPanel import ControlPanel
from NPCLogger import NPCBlank, NPCLog, NPCLogTitleContext

from FedFMK.FedClient import Client
from FedFMK.FedUtils import FedDataShell
from FedFMK.FedAUX.nn import ClassifyShell

from Dataset.utils import NPCDataset
from Models.TinyModels import LogisticRegression

class LogisticRegressionDataShell(NPCDataset):
    def __init__(self, pos_data , neg_data , feature_extractor , data_source = "") -> None:
        super(LogisticRegressionDataShell,self).__init__(data_source)
        
        self.dataset = [(pos_data[idx][0],1) for idx in range(len(pos_data))]
        self.dataset+= [(neg_data[idx][0],0) for idx in range(len(neg_data))]
        self.feature_extractor = feature_extractor
        self.feature_extractor.to(ControlPanel.device)
        # self.read()

    def read(self):
        new_dataset = []
        for idx in range(len(self.dataset)):
            new_dataset.append(self.__convert_item__(idx))
        self.dataset = new_dataset

    def __convert_item__(self, idx):
        data,label = self.dataset[idx]
        _,p = self.feature_extractor(data.unsqueeze(0).to(ControlPanel.device))
        data = p.detach().squeeze(0).cpu()
        return data,torch.tensor([label],dtype=torch.float32)

    def __getitem__(self, idx):
        return self.__convert_item__(idx)
        return self.dataset[idx]

    @staticmethod
    def cmp(o,y):
        return (o > 0.5) == (y > 0.5)

class FedAUXClient(Client):
    def __init__(self, train_dataset: FedDataShell, test_datset: FedDataShell = None  , split=False, split_ration=0.5) -> None:
        super(FedAUXClient,self).__init__(train_dataset, test_datset, split, split_ration)

    def DownLoadModel(self,model:ClassifyShell,modelShell = None):
        # self.model = NNInterFace.CloneModule(model,modelShell)
        self.model = deepcopy(model)

    def DownloadAUXData(self,negative_data , auxdata_distill , feature_extractor):
        self.negative_data = negative_data
        self.auxdata_distill = auxdata_distill
        self.feature_extractor = feature_extractor

    def ComputeScore(self):
        with NPCLogTitleContext("client %d report"%self.id):
            NPCLog("computing score ...")
            netname = "10Client_%d_LogisticRegression0.01"%self.id
            self.lgregr = utils.CheckModel(netname)
            
            if self.lgregr == None:
                dataset = LogisticRegressionDataShell(self.train_dataset,self.negative_data,self.feature_extractor)
                self.lgregr = LogisticRegression(128,1)
                self.lgregr.lossf = torch.nn.BCELoss()
                self.lgregr.name = netname
                self.lgregr.compare_output_label = LogisticRegressionDataShell.cmp
                dataset.feature_extractor.to(ControlPanel.device)
                if not ControlPanel.Debug:
                    self.lgregr.Train(dataset,None,8,save = True,endACU=99,use_train_acuv=True)

            ####################差分隐私###########################



            ######################################################
            self.lgregr.to(ControlPanel.device)
            self.lgregr.compare_output_label = LogisticRegressionDataShell.cmp
            self.auxdata_distill_scores = self.lgregr.TestDataset(
                LogisticRegressionDataShell(self.auxdata_distill,[],self.feature_extractor),lambda x:x
            )
            self.auxdata_distill_scores = torch.tensor(self.auxdata_distill_scores).T.squeeze(0).tolist()
            self.lgregr.cpu()
            NPCLog("auxdata_distill_mean_scores : ",mean(self.auxdata_distill_scores))

    def CollectAuxDataLabel(self):
        with NPCLogTitleContext("client %d report"%self.id):
            NPCLog("compute auxdata distill labels")
            self.auxdata_distill_labels = self.model.TestDataset(self.auxdata_distill,lambda x:x)

    def TrainModel(self, echo=5, test=False):
        with NPCLogTitleContext("client %d report"%self.id):
            self.model.train_feature_extractor = True
            self.model.distill_mode = False
            self.model.Train(self.train_dataset,self.test_datset if test else None,echo=echo,batch_echo=100,save=False,lr=self.model.lr,early_stop=False)
