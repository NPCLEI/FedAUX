import math
import utils

from numpy import random
from statistics import mean
from .FedUtils import FedUtils
from .FedUtils import FedDataShell
from .FedClient import Client,Client_
from ControlPanel import ControlPanel
from NPCLogger import NPCLog,NPCBlank,NPCLogTitleContext,NPCLine

class Server:    
    
    def __init__(self,initModelFunc,global_test_data = None,numpkclient = 10 , CLIENTCLASS = Client) -> None:
        self.initModelFunc = initModelFunc
        self.__shuffle_clients_idx__ = []
        self.numpkclient = numpkclient
        self.last_parter_num = numpkclient
        self.global_model = initModelFunc()
        self.CLIENTCLASS = CLIENTCLASS
        self.clients = []
        self.min_num_pick_client = 1
        self.statistics = []
        self.global_test_data = global_test_data

    def __len__(self):
        return len(self.clients)

    def __getitem__(self,idx):
        return self.clients[idx]


    def ResgisterClientFromDatasets(self,train_dataset):
        for dataset in train_dataset:
            if len(dataset) == 0:
                continue
            self.ResgisterClient(self.CLIENTCLASS(dataset))

    def ResgisterClient(self,client):
        client.id = len(self.clients)
        self.clients.append(client)
        self.__shuffle_clients_idx__.append(client.id)

    @property
    def last_picked_clients_ids(self):
        if self.last_parter_num < 0:
            return self.__shuffle_clients_idx__
        return self.__shuffle_clients_idx__[:self.last_parter_num]

    def RandomPickClients(self,seed = 42):
        # random.seed(seed)
        if self.numpkclient <= 0:
            return self.__shuffle_clients_idx__
        elif 0 < self.numpkclient < 1:
            self.last_parter_num = math.ceil(len(self.clients) * self.numpkclient)
        elif type(self.numpkclient) == type((10,20)):
            l,r = self.numpkclient
            cur_parter_num =  0
            while cur_parter_num < self.min_num_pick_client:
                cur_parter_num =  (len(self.clients) * random.randint(l,r)) // 100
            self.last_parter_num = cur_parter_num
        elif self.numpkclient >= 1:
            self.last_parter_num = self.numpkclient
        else:
            raise Exception("numpkclient is ",self.numpkclient)
        
        if self.last_parter_num  == 0:
            self.last_parter_num = self.min_num_pick_client

        random.shuffle(self.__shuffle_clients_idx__)
        return self.__shuffle_clients_idx__[:self.last_parter_num]

    def MergeClientsModel(self,clients_idxes):
        return FedUtils.FedAVG(
            [self.clients[client_id].model for client_id in clients_idxes],
            [len(self.clients[client_id].train_dataset) for client_id in clients_idxes],
            globalShell = self.initModelFunc()
        )

    def Setting(self,local_updating_step = 20,mini_batch = 32,loss_record = True):
        self.local_updating_step = local_updating_step
        self.mini_batch = mini_batch
        self.loss_record = loss_record

    def Statistics(self,comicn,client_id,all_test_batch_losses,testacuv):
        self.statistics.append({
            "communication times":comicn,
            "record id":client_id,
            "all_test_batch_losses":all_test_batch_losses,
            "testacuv":testacuv
        })

    def PackModels(self):
        return self.global_model

    def Train(self,max_comicn_num = 50, global_test_data = None):
        global_test_data ,testacuv= self.global_test_data , -1
        ControlPanel.batch_size = self.mini_batch
        for comicn in range(max_comicn_num):
            with NPCLogTitleContext("server %dth report"%comicn):
                NPCLog("Number of current communication is: ",comicn," times.")
                clients_idxes = self.RandomPickClients()
                NPCLog("Picked id of Clients:",clients_idxes)
                NPCBlank()
                if global_test_data != None and not ControlPanel.Debug:
                    NPCLine('#')
                    NPCLog("Server is testing the global model")
                    testacuv = self.global_model.Test(global_test_data,True)
                    self.Statistics(comicn,-1,self.global_model.all_test_batch_losses,testacuv)
                    NPCLog("Server tested.",title='')
                NPCLine('#')
                NPCBlank()
                NPCLine()
                for client_id in clients_idxes:
                    client = self.clients[client_id]
                    client.cur_server_comicn = comicn
                    NPCLog("Client ",client.id," downing global model...",end="")
                    # client.DownLoadModel(self.global_model,self.initModelFunc())
                    client.DownLoadModel(self.PackModels())
                    NPCLog("done.",title="")
                    if global_test_data == None:
                        NPCLog("Client %s is testing the global model"%client.id)
                        testacuv = client.TestModel()
                        self.Statistics(comicn,client.id,client.model.train_mean_loss,testacuv)
                        NPCLog(self.statistics[-1])
                        NPCLog("Client tested.")
                    NPCLog("Client training the local model")
                    client.TrainModel(self.local_updating_step)
                    if global_test_data != None:
                        self.Statistics(comicn,client.id,client.model.train_mean_loss,testacuv)
                    NPCLog("Client %s finished training the model."%client.id)
                    NPCLine()
                    
                # merge models
                self.global_model = self.MergeClientsModel(clients_idxes)
                for client_id in clients_idxes:
                    self.clients[client_id].EndCommuc()
            utils.CheckModel('%s_statistics'%(str(type(self))[8:-2]),lambda:self.statistics,retrain=True)
            # utils.SaveJson(loss,"loss_record")
        return self.statistics

    @staticmethod
    def PlotStatistics(statistics,show = True,label='',mode = "test acuv",xticstep = 50):
        record = {}
        for ls in statistics:
            cn = ls[0]
            if cn not in record:
                record.setdefault(cn,[])
            if mode == "loss":
                record[cn].append(mean(ls[2]))
            elif mode == "client train loss" or mode == "ctl":
                if ls[1] < 0:
                    continue
                record[cn].append(mean(ls[2]))
            else:
                record[cn].append(ls[3])
        xx = []
        yy = []
        for cn in record:
            xx.append(cn)
            yy.append(mean(record[cn]))
        xx = [i*10 for i in range(len(yy))]
        print(xx)

        from matplotlib import pyplot as plt
        
        plt.xticks([0,50,100,150,200])
        plt.plot(xx,yy,label=label)
        plt.legend()
        if show:
            plt.show()



if __name__ == "__main__":
    server = Server(lambda:1)
    for i in range(20):
        server.ResgisterClient(Client_())
    server.Train()