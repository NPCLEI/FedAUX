
import torch
from FedFMK.FedAUX.nn import ClassifyShell
from FedFMK.FedUtils import FedUtils
from NPCLogger import NPCLog
import utils

from Models.nn import NPCModule, NetStatistics
from .Client import FedAUXClient
from Dataset.utils import NPCDataset
from ..FedServer import Server as Server

class AUXDistillDatasetShell(NPCDataset):
    def __init__(self,server:Server) -> None:
        NPCDataset.__init__(self,None)
        self.clients_idx = server.last_picked_clients_ids
        self.server = server

    def __len__(self):
        return len(self.server.distillation_data)

    def __getitem__(self, idx):

        y_scores = []
        weights = 0
        for client_id in self.clients_idx:
            client = self.server.clients[client_id]
            weight = client.auxdata_distill_scores[idx]
            y_score = weight * torch.tensor(client.auxdata_distill_labels[idx]) 

            y_scores.append(y_score.unsqueeze(0))
            weights += weight
        
        y_scores = torch.cat(y_scores,dim=0)
        y_scores = y_scores.sum(dim=0)

        return self.server.distillation_data[idx][0],torch.softmax(y_scores / weights,0)

class FedAUXServer(Server):
    def __init__(self,initModelFunc:NPCModule,aux_data:NPCDataset = None,negative_ratio = 0.2,numpkclient = 10,global_test_data = None) -> None:
        Server.__init__(self,initModelFunc,numpkclient)
        self.auxdata = aux_data
        if self.auxdata != None:
            self.SplitAuxData(negative_ratio)
        self.global_model = initModelFunc()
        self.CLIENTCLASS = FedAUXClient
        self.global_test_data = global_test_data

    def SplitAuxData(self,negative_ratio = 0.2):
        self.distillation_data,self.negative_data = self.auxdata.split(negative_ratio)
        NPCLog("cut distill data : ",len(self.distillation_data),len(self.negative_data))

    def LoadPretrainedModel(self,model):
        # if type(model) == type(""):
        #     self.preTrainedModel = utils.CheckModel(model)
        # else:
        self.preTrainedModel = model

    def ComputeScore(self):
        for client in self.clients:
            client.DownloadAUXData(self.negative_data,self.distillation_data,self.preTrainedModel)
            client.ComputeScore()
        ###############collect scores############################

    def MergeClientsModel(self, clients_idxes):
        self.CollectClientsAuxDataLabel(clients_idxes)
        self.global_model = FedUtils.FedAVG(
            [self.clients[client_id].model for client_id in clients_idxes],
            [len(self.clients[client_id].train_dataset) for client_id in clients_idxes],
            globalShell = self.initModelFunc()
        )
        testacuv = self.global_model.Test(self.global_test_data)
        self.Statistics(-1,-1,self.global_model.all_test_batch_losses,testacuv)
        return self.DistillModule()

    def CollectClientsAuxDataLabel(self, clients_idxes):
        for clients_idx in clients_idxes:
            self.clients[clients_idx].CollectAuxDataLabel()

    def DistillModule(self):
        # debug
        self.global_model.clear_states()
        self.global_model.distill_mode = True
        if self.global_model.name == "":
            self.global_model.name = "DistillTempModel_%s"%self.global_model.Name
        self.global_model.cmp = NetStatistics.compare_label_onehot_vector
        self.global_model.test_cmp = NetStatistics.compare_label_idx
        # self.global_model = self.global_model.Train(
        #     AUXDistillDatasetShell(self),
        #     self.global_test_data,echo = 3,save = False,early_stop=False,use_train_acuv=True,
        #     lossf = torch.nn.KLDivLoss(reduction='mean'),save_acu_lower_bound= 26
        # )
        self.global_model = self.global_model.Train(
            AUXDistillDatasetShell(self),echo = 1,save = False,early_stop=False,
            lossf = torch.nn.KLDivLoss(reduction='mean')
        )
        self.global_model.loss_func = torch.nn.CrossEntropyLoss()
        self.global_model.cmp = NetStatistics.compare_label_idx
        self.global_model.distill_mode = False
        return self.global_model

    def Train(self, max_comicn_num=50):
        return super().Train(max_comicn_num)