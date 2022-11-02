
import sys
import torch
sys.path.append(r'./')

import utils

from NPCLogger import NPCLog
from Models.nn import NetStatistics
from Dataset.AGNews import AGNewsCsv
from ControlPanel import ControlPanel
from Dataset.BookCorpus import BookCorpus

from FedFMK.FedServer import Server
from FedFMK.FedAUX.nn import ClassifyShell
from FedFMK.FedAUX.Server import FedAUXServer
from FedFMK.FedAUX.Client import FedAUXClient
from FedFMK.FedUtils import FedDataDistributer,FedUtils

from Models.TinyModules.TinyBert.tiny_bert import get_tiny_bert


def initModel():
    net = ClassifyShell(get_tiny_bert(),4)
    net.compare_output_label = NetStatistics.compare_label_idx
    return net

if __name__ == "__main__":
    ControlPanel.Debug = False
    ControlPanel.WorkSpace("Pickle")

    data_path = r"E:\Dataset\NLP-Other\AGNews"

    client_num = 10
    distributeMatrix = utils.CheckModel('%d_distributeMatrix_1.0'%client_num,lambda:FedUtils.DistributeMatrix(client_num,4,1.0,seed=None))

    dataset = utils.CheckModel("AGNewsCsv_train",lambda:AGNewsCsv('%s/train.csv'%(data_path)))
    global_test_data = utils.CheckModel("AGNewsCsv_test",lambda:AGNewsCsv('%s/test.csv'%(data_path)))
    if ControlPanel.Debug:
        dataset.limit(dict([(cls,100) for cls in dataset.label_idx_map]))
    NPCLog("train: ",len(dataset)," test: ",len(global_test_data),". num of cls is ",dataset.cls_num)

    aux_data = utils.CheckModel("BookCorpus2200",lambda:BookCorpus(r'E:\Dataset\NLP-Other\BookCorpus\epubtxt',100))
    if ControlPanel.Debug:
        aux_data.dataset = aux_data.dataset[:400]
    NPCLog("num of aux_data: ",len(aux_data))

    distributer = FedDataDistributer(dataset,distributeMatrix)
    distributer.init(lambda item:item[1].item())
    
    # start Train
    commu_times,local_updating_step = 20,10 if ControlPanel.Debug else 10

    server = FedAUXServer(initModel,aux_data,numpkclient=-1,global_test_data=global_test_data)
    server.ResgisterClientFromDatasets(distributer.Distribute())
    server.Setting(
        local_updating_step = local_updating_step,
        mini_batch = 600,
        loss_record = True
    )

    server.LoadPretrainedModel(get_tiny_bert())
    server.ComputeScore()
    statistics = server.Train(commu_times)

    utils.CheckModel('last_avg_statistics',lambda:statistics,retrain=True)
    Server.PlotStatistics(statistics,False,mode='ctl',label='avg')

    utils.CheckModel('avg_global_model',lambda:server.global_model)