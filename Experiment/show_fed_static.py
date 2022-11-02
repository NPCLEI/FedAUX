
import math
from statistics import mean
import sys
sys.path.append(r'C:\Users\NPC\Desktop\MyGithub\FedEffAUX')

import torch
import utils
import numpy as np

from NPCLogger import NPCLog
from Dataset.AGNews import AGNewsCsv
from ControlPanel import ControlPanel
from Dataset.BookCorpus import BookCorpus
from scipy.signal import savgol_filter

from FedFMK.FedAUX.nn import ClassifyShell
from FedFMK.FedAUX.Server import FedAUXServer
from FedFMK.FedAUX.Client import LogisticRegressionDataShell
from FedFMK.FedUtils import FedDataDistributer,FedUtils

from Models.TinyModules.TinyBert.tiny_bert import get_tiny_bert
from Models.TinyModels import LogisticRegression

    # model = LogisticRegression(128,1)
    # aux_data = BookCorpus(r'E:\Dataset\NLP-Other\BookCorpus\epubtxt')

    # aux_data = LogisticRegressionDataShell(aux_data,[],get_tiny_bert(r'./Models/TinyModules/TinyBert'))

    # res = model.TestDataset(aux_data,lambda x:x)

    # print(res)
    # print(torch.sum(res))
    # print(res.numel() - torch.sum(res))

if __name__ == "__main__":
    
    # s = np.random.dirichlet(np.ones(4)*0.01, 10)

    # FedUtils.PlotDistributeMatrix(FedUtils.DistributeMatrix(10,4,0.01),1,True)

    from matplotlib import pyplot as plt
    # plt.rc('font',family='Times New Roman')
    
    # mtx = utils.CheckModel("/fed_gen_statistics/dmtx_exgen_20_26_0.05")
    # mtx = utils.CheckModel("/fed_gen0.01_statistics/dmtx_exgen_20_26_0.01")
    # FedUtils.PlotDistributeMatrix(mtx,2,True)


    def printItem(item):
        for key in item:
            if key != 'all_test_batch_losses':
                print(key,item[key],end='\t')
        print()

    def plot(name='fed_gen_statistics/temp_gen_statistics',label = 'fedgen',c=None,mode = 'g',smooth = True,ctk = False):
        statistics = utils.CheckModel(name)
        if  c == None:
            if 'gen' in label:
                c = 'g'
            if 'avg' in label:
                c = 'r'
            if 'prox' in label:
                c = 'b'

        gb,stk,bgb = [],[],[]
        for idx,item in enumerate(statistics):
            # if item["communication times"] != -1:
            if mode == 'g' and item["record id"] == -1 :
                if item['communication times'] != -1:
                    gb.append(float(item["testacuv"]))
                    printItem(item)
                else:
                    bgb.append(float(item["testacuv"]))
            if mode == 'cs' and item["record id"] != -1:
                stk.append(float(item["testacuv"]))
            elif mode == 'cs' and item["record id"] == -1 and len(stk) > 0:
                gb.append(mean(stk))
                stk = []
        def _plot(gb):
            print(len(gb))
            plt.plot([_ for _ in range(len(gb))],[max(gb) for _ in range(len(gb))],linestyle='dotted',c=c)
            plt.plot([_ for _ in range(len(gb))],gb,c=c,alpha=0.2 if smooth else 1)
            if smooth:
                # window_size = math.ceil(len(gb) / 3) + 2
                gb = savgol_filter(gb, 20 , 5) # window size 51, polynomial order 3
                # gb = savgol_filter(gb, 30 , 3) # window size 51, polynomial order 3
                plt.plot([_ for _ in range(len(gb))],gb,c=c,label = label)
        if not ctk:
            _plot(gb)
        else:
            label = 'before distill'+label
            _plot(bgb)
    # plot(name='fed_gen_statistics/final_gen_statistics',label = 'fedgen',c = 'g')
    # plot(name='fed_gen_statistics/final_avg_statistics',label = 'fedavg',c='r')
    # plot(name='fed_gen_statistics/final_prox_statistics',label = 'prox',c = 'b')

    smooth = True

    # plot(name='final_gen_5_0.01_statistics/temp_gen_statistics',label = 'fedgen',c = 'g',smooth=smooth)
    # plot(name='final_gen_5_0.01_statistics/final_gen_statistics',label = 'fedgen',c = 'g',smooth=smooth)
    # # plot(name='final_gen_5_0.01_statistics/final_prox_statistics',label = 'fedprox',c='b',smooth=smooth)
    # plot(name='final_gen_5_0.01_statistics/final_avg_statistics',label = 'fedavg',c='r',smooth=smooth)

    # plot(name='final_gen_5_0.01_statistics/temp_gen_statistics',label = 'fedgen',c = 'g',smooth=smooth)
    # plot(name='real_final_gen/final_prox_statistics',label = 'fedprox',c='b',smooth=smooth)
    
    plt.figure(1)

    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Amazon 16,alpha = 0.05")

    folder = 'limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5'
    # mtx = utils.CheckModel('%s/dmtx_exgen_20_16_0.05'%folder)
    # FedUtils.PlotDistributeMatrix(mtx,2,True)
    plt.plot([_ for _ in range(50)],[87.56319 for _ in range(50)],label = 'center train',c='black')
    plot(name='limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5/final_avg_statistics',label = 'fedavg',smooth=smooth)
    plot(name='limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5/final_prox_statistics',label = 'fedprox',smooth=smooth)
    plot(name='limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5/final_gen_statistics',label = 'fedgen',c = 'DeepSkyBlue',smooth=smooth)
    plot(name='limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5/final_aux_statistics',label = 'fedaux',smooth=smooth,ctk=True)
    plot(name='limit_500_noiid_dirichlet_dmtx_exgen_20_16_0.05_1e-5/final_adavg_statistics',label = 'adversarial domain fedavg',c = 'purple',smooth=smooth)
    
    # plot(name='%s/final_aux_statistics'%folder,label = 'fedaux',c = 'g',smooth=smooth,ctk=True)
    # plot(name='2018Amz/PNIID/final_avg_statistics',label = 'fedavg',smooth=smooth)
    # plot(name='2018Amz/PNIID/final_aux_statistics',label = 'fedaux',c = 'g',smooth=smooth,ctk=True)
    # plot(name='2018Amz/PNIID/final_prox_statistics',label = 'fedprox',smooth=smooth)
    # plot(name='2018Amz/PNIID/final_gen_statistics',label = 'fedgen',c = 'purple',smooth=smooth)
    
    # plot(name='FINAL_GEN200/final_prox_statistics',label = 'fedprox',smooth=smooth)
    # plot(name='FINAL_GEN200/temp_gen_statistics',label = 'gen',c = 'purple',smooth=smooth)
    # plot(name='FINAL_GEN200/ttemp_gen_statistics',label = 'tg',smooth=False)
    # plot(name='FINAL_GEN200/100_gen_statistics',label = 'gen',smooth=smooth)
    
    # plot(name='gen20_50/final_gen_statistics',label = 'fedgen' ,smooth=smooth)
    # plot(name='gen20_50/final_prox_statistics',label = 'prox' ,smooth=smooth)
    # plot(name='gen_20_50/final_gen20g_statistics',label = 'tfedgen',smooth=smooth)
    # plot(name='real_final_gen/final_gen_statistics',label = 'fedgen',c='g',smooth=smooth)
    # plot(name='gen40/final_gen_statistics',label = 'fedgen',c = 'g',smooth=smooth)

    # plt.xticks([_ for _ in range(len(gb))])
    plt.legend()
    plt.show()
