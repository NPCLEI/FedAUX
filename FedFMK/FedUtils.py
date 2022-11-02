import sys
sys.path.append(r'C:\Users\NPC\Desktop\MyGithub\FedPick')

import torch
import numpy as np
from tqdm import tqdm
from math import ceil
from NPCLogger import NPCLog
from numpy.random import shuffle
from ControlPanel import ControlPanel

class FedUtils:

    @staticmethod
    def copyModel(or_model,tar_model):

        for or_param,tar_param in zip(or_model.parameters(True),tar_model.parameters(True)):
            tar_param.data =  torch.clone(or_param.data)

        return tar_model

    @staticmethod
    def closeGrad(tensor:torch.Tensor):
        res = tensor.clone()
        return res

    @staticmethod
    def FedAVG(models,sampleNums = None,globalShell = None):
        numModels = len(models)
        assert sampleNums == None or len(sampleNums) == numModels,"样本参数数量与聚合模型数量不一致"
        if numModels == 1:
            return models[0]
        if sampleNums == None:
            avgWeights = [1/numModels for _ in range(numModels)]
        else:
            numofsample = sum(sampleNums)
            avgWeights = [num/numofsample for num in sampleNums]
        
        if globalShell == None:
            globalModel = type(models[0])()
        else:
            globalModel = globalShell
        # NPCLog(globalModel)

        NPCLog("fedavg modouls:")
        for params in tqdm(zip(globalModel.parameters(True),*[model.parameters(True) for model in models])):

            params[0].data = params[1].data * avgWeights[0]
            for idx in range(2,len(params)):
                params[0].data += params[idx].data * avgWeights[idx-1]

        return globalModel

    @staticmethod
    def rebuildFrom(svd_params,modelshell):
        #rebuild the model
        NPCLog("rebuild modouls:")
        for target_param,param in tqdm(zip(modelshell.parameters(True),svd_params)):
            if len(param) == 3:
                param = FedUtils.rebuildSVD(param)
            
            target_param.data = param
        return modelshell

    @staticmethod
    def rebuildSVD(usv):
        with torch.no_grad():
            u,s,v = usv
            u,s,v = u.to(ControlPanel.device),s.to(ControlPanel.device),v.to(ControlPanel.device)
            res = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))
            return res.cpu()

    @staticmethod
    def svdModel(model,below = 0.95):
        model.to(ControlPanel.device)
        with torch.no_grad():
            params = []
            NPCLog('svd model...')
            for param in tqdm(model.parameters(True)):
                if len(param.shape) == 1:
                    params.append(param)
                    continue
                u,g,v = torch.svd(param)
                #g like (h , w)
                below = torch.sum(torch.pow(g,2)) * below
                for k in range(len(g)):
                    e = torch.sum(torch.pow(g[:k],2))
                    if e > below:
                        break
                # It is fun
                #########################################
                ##########Do not forget this g[k:] = 0!#################
                g[k:] = 0 
                params.append([u.cpu(),g.cpu(),v.cpu()])
            return params

    @staticmethod
    def Distribute(data,):
        pass



    @staticmethod
    def DistributeMatrix(clinets_num,label_num,alpha = 1, limit_ratio = -1,seed = None ,method = 'dirichlet'):
        if seed != None:
            np.random.seed(seed)

        if limit_ratio < 0:
            return torch.tensor(np.random.dirichlet(np.ones(label_num)*alpha, clinets_num)).tolist()
            
        res = [[0 for _ in range(label_num)] for _ in range(clinets_num)]
        for cls in range(label_num):
            points = np.random.dirichlet(np.repeat(alpha, clinets_num))
            for pidx in range(clinets_num):
                while points[pidx] > limit_ratio:
                    cut_value = np.random.randint(int(limit_ratio * 50),int(limit_ratio * 100)) / 100
                    tr_pidx = np.random.randint(0,clinets_num)
                    if tr_pidx == pidx and (points[tr_pidx] + cut_value) > limit_ratio + 0.5:
                        continue
                    points[tr_pidx] += cut_value
                    points[pidx] -= cut_value
                    

            for cn in range(clinets_num):
                res[cn][cls] = points[cn]

        return res

    @staticmethod
    def PlotDistributeMatrix(dmtx,zoom = 1,show = False):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        plt.grid()

        points = dmtx
        clients_num,label_num = len(dmtx),len(dmtx[0])
        for cls in range(label_num):
            for idx in range(clients_num):
                NPCLog("%2.3f\t"%points[idx][cls],end='',title="")
                ax.add_patch(plt.Circle((idx, cls), points[idx][cls] * zoom , color='b'))
                ax.add_patch(plt.Circle((idx, cls), points[idx][cls] * zoom , color='black',fill=False))
            NPCLog(title="")

        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()   
        if show:
            plt.show()


from Dataset.utils import NPCDataset as NPCDataset

class FedDataShell(NPCDataset):
    def __init__(self,distributer,datamap,getLabel = lambda item:item[1]) -> None:
        super(FedDataShell, self).__init__("")
        self.getLabel = getLabel
        self.distributer = distributer
        self.datamap = datamap
        self.dataset = self

    def shuffle(self):
        return shuffle(self.datamap)

    def PrintLabelDistribution(self):
        clsMap = {}
        for idx in range(len(self)):
            item = self[idx]
            label = self.getLabel(item)
            if label not in clsMap:
                clsMap.setdefault(label,[])
            clsMap[label].append(idx)
        for label in clsMap:
            print(label,len(clsMap[label]))

    def Split(self,test_ratio = 0.2):
        clsMap = {}
        for idx in range(len(self)):
            item = self[idx]
            label = self.getLabel(item)
            if label not in clsMap:
                clsMap.setdefault(label,[])
            clsMap[label].append(idx)
        train,test = FedDataShell(self.distributer,[]),FedDataShell(self.distributer,[])
        for label in clsMap:
            mid = int(len(clsMap[label])*(1-test_ratio))
            train.datamap += clsMap[label][:mid]
            test.datamap += clsMap[label][mid:]
        return train,test

    def __getitem__(self, idx):
        return self.distributer.dataset[self.datamap[idx]]

    def __len__(self):
        return len(self.datamap)

class FedDataDistributer:
    class MapItem:
        def __init__(self) -> None:
            self.data_idxes = []
            self.dp = 0

        def append(self,value):
            self.data_idxes.append(value)

        def pop(self):
            res = self.data_idxes[self.dp]
            self.dp += 1
            return res
        
        def popList(self,times):
            res = []
            for i in range(times):
                if self.dp + i >= len(self.data_idxes):
                    break
                res.append(self.pop())
            return res

        def __str__(self) -> str:
            return str(self.data_idxes)

    """
        create a map:map the original data -> clients data
    """
    def __init__(self,data,distributeMatrix,shellGetLabelFunc = lambda item:item[1]) -> None:
        self.dataset = data
        self.dtrmat = distributeMatrix
        clients_num,label_num = len(distributeMatrix),len(distributeMatrix[0])
        self.clients_num = clients_num
        self.label_num = label_num
        self.clsMap = dict([
            (cls,FedDataDistributer.MapItem()) for cls in range(label_num)
        ])
        self.init_flag = False
        self.shellGetLabelFunc = shellGetLabelFunc

    def SupplementsData(self,data,catFunc,getLabel = lambda item:item[0],shuffle = True):
        self.dataset = catFunc(self.dataset + data)
        for idx in range(len(self.dataset),len(self.dataset) + len(data)):
            item = data[idx]
            label = getLabel(item)
            self.clsMap[label].append(idx)
        if shuffle:
            for label in self.clsMap:
                np.random.shuffle(self.clsMap[label].data_idxes)

    @staticmethod
    def collectDataByLabel(data,getLabel):
        clsMap = {}
        # (cls,FedDataDistributer.MapItem())
        for idx in range(len(data)):
            item = data[idx]
            label = getLabel(item)
            if label not in clsMap:
                clsMap.setdefault(label,FedDataDistributer.MapItem())    
            clsMap[label].append(idx)
        return clsMap  

    def PrintClsMap(self):
        for key in self.clsMap:
            print(key,self.clsMap[key])
            print()

    def init(self,getLabel = lambda item:item[0],getItem = lambda data,idx:data[idx]):
        for idx in range(len(self.dataset)):
            item = getItem(self.dataset,idx)
            label = getLabel(item)
            self.clsMap[label].append(idx)
        self.init_flag = True

    def Distribute(self):
        import math
        if not self.init_flag:
            self.init()
        shelles = []

        ##########根据迪利克雷分布领取数据,因为整数取向下取整,保证数据不会在前几个客户端被优先分配。##############
        ##########故此步过后会有数据剩余################
        for cln in range(self.clients_num):
            shell = FedDataShell(self,[],getLabel=self.shellGetLabelFunc)
            for label_idx,clsp in zip(range(self.label_num),self.dtrmat[cln]):
                cls_sample_num = int(clsp * len(self.clsMap[label_idx].data_idxes))
                shell.datamap += self.clsMap[label_idx].popList(cls_sample_num)
            shelles.append(shell)

        ########补充没有被添加的数据###############
        for cln in range(self.clients_num):
            shell = shelles[cln]
            for label_idx,clsp in zip(range(self.label_num),self.dtrmat[cln]):
                cls_sample_num = ceil(clsp * len(self.clsMap[label_idx].data_idxes))
                shell.datamap += self.clsMap[label_idx].popList(cls_sample_num)
            np.random.shuffle(shell.datamap)
        return shelles

if __name__ == "__main__":
    fakedata = []
    for cls in range(10):
        for _ in range(1000):
            fakedata.append((cls,"%d-%d"%(cls,_)))
    dter = FedDataDistributer(fakedata,FedUtils.DistributeMatrix(20,10,0.16,seed=24))
    dter.init()
    # dter.PrintClsMap()
    shelles = dter.Distribute()
    summ = 0
    for shell in shelles:
        summ += len(shell.datamap)
        print(shell.datamap)
        print(len(shell.datamap))
    print(summ)
    FedUtils.PlotDistributeMatrix(dter.dtrmat)