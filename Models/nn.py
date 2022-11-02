import sys
sys.path.append("../")

import math
import torch
import utils
from ControlPanel import ControlPanel
import torch.nn as nn

from numpy.random import randint
from statistics import mean
from NPCLogger import NPCLine, NPCLog
from torch.utils.data import DataLoader

from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class ReverseGradient(nn.Module):
    def forward(self,x , alpha = 0.5):
        return ReverseLayerF.apply(x, alpha)


class NetStatistics:
    @staticmethod
    def getPrgl(len_datset):
        return math.ceil(len_datset/ControlPanel.batch_size)

    @staticmethod
    def compare_label_idx(o,label_idx):
        if o.size(0) == 0:
            return torch.tensor([True]) 
        return torch.argmax(o,dim=1)==label_idx

    @staticmethod
    def compare_label_onehot_vector(o,label_vector):
        if o.size(0) == 0:
            return torch.tensor([True]) 
        return torch.argmax(o,dim=1)==torch.argmax(label_vector,dim=1)

    @staticmethod
    def PrintBatchInfo(acu_dict,echo_count,batch_count,prgl,all_batch_losses = [-1]):
        if (acu_dict[True]+acu_dict[False]) != 0:
            acuvalue = 100*acu_dict[True]/(acu_dict[True]+acu_dict[False])
        else:
            acuvalue = -1
        NPCLog( " echo:",echo_count,
                "(%d/%d[%2.2f%%])"%(batch_count,prgl,100*batch_count/prgl),
                " loss:",all_batch_losses[-1],
                " batch acu:",acu_dict,
                " acuv:",acuvalue)
        return acuvalue

    @staticmethod
    def get_acuv(acu_dict):
        if (acu_dict[True]+acu_dict[False]) != 0:
            acuvalue = 100*acu_dict[True]/(acu_dict[True]+acu_dict[False])
        else:
            acuvalue = -1
        return acuvalue

    @staticmethod
    def output2label(o):
        return torch.argmax(o,dim=1)

class ModuleUtils:
    @staticmethod
    def CloneModule(model,modelShell = None):
        if modelShell == None:
            globalModel = type(model)()
        else:
            globalModel = modelShell
        # NPCLog(globalModel)

        for params in zip(globalModel.parameters(True),model.parameters(True)):
            params[0].data = torch.clone(params[1].data)
        
        return globalModel

# 定义网络结构
class NPCModule(nn.Module):

    CELoss = torch.nn.CrossEntropyLoss()
    BCELoss = torch.nn.BCELoss()
    MSELoss = torch.nn.MSELoss()
    _KLD = torch.nn.KLDivLoss(reduction='batchmean')

    @staticmethod
    def KLD(output,target):
        return NPCModule._KLD(output.log_softmax(1),target)
        
    def __init__(self,name = ''):
        super(NPCModule,self).__init__()
        self.name = name
        self.userCustom = False
        self.best_save_acuv = -1
        self.__compare__ = NetStatistics.compare_label_idx
        self.lossf = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam
        self.train_acuv_history = []
        self.test_cmp = self.compare_output_label

    def compare_output_label(self,o,p):
        try:
            return self.__compare__(o,p)
        except:
            if self.__compare__ == NetStatistics.compare_label_idx:
                self.__compare__ = NetStatistics.compare_label_onehot_vector
            else:
                self.__compare__ = NetStatistics.compare_label_idx
            return self.__compare__(o,p)

    def clear_states(self):
        self.train_acuv_history = []
        self.best_save_acuv = -1
        self.all_test_batch_losses = []
        self.batch_acu_statistics = {True:0,False:0}

    @property
    def Name(self):
        if self.name == "":
            self.name = str(type(self))[8:-2] + str(randint(10000,99999))
        return self.name

    def __SaveBest__(self,acuv):
        if acuv > self.best_save_acuv:
            self.best_save_acuv = acuv
            self.save(self.name)
            return True
        return False

    def save(self,name = ''):
        import pickle
        f_name = "%s/%s.pickle"%(ControlPanel.pickle_path,self.Name)
        self.cpu()
        with open(f_name, 'wb+') as net_file:
            pickle.dump(self,net_file)
        
        self.to(ControlPanel.device)
        return self.Name

    def TestDataset(net,dataset,output2label = NetStatistics.output2label) -> list:
        if not hasattr(dataset,"collate_func"):
            dataset.collate_func = None
        device = ControlPanel.device
        loader = DataLoader(dataset, batch_size = ControlPanel.batch_size, shuffle=False,collate_fn=dataset.collate_func)
        prgl = math.ceil(len(dataset)/ControlPanel.batch_size)
        net.to(ControlPanel.device)
        net.eval()

        net.all_test_batch_losses = [0]

        NPCLog("Collect dataset predict result.Testing... ",end = "")
        net.test_acu = {True:0,False:0}
        with torch.no_grad():
            result = []
            batch_count,lastprg = 0,0
            for item in loader:
                x,y = item[0].to(device),item[1].to(device)
                o = net(x)

                batch_labels = output2label(o)
                if type(batch_labels) == type([]):
                    result += batch_labels 
                else:
                    result += batch_labels.tolist()

                cmp_res = net.compare_output_label(o,y)
                net.test_acu[True] += torch.sum(cmp_res).item()
                net.test_acu[False] += abs(cmp_res.numel() - net.test_acu[True])

                batch_count += 1

                prgs = 100*batch_count/prgl
                if (prgs - lastprg) >= 9.7:
                    NPCLog("%2.1f%%. "%prgs,end=" ",title=False,flush = True)
                    lastprg = prgs

                x,y = x.cpu(),y.cpu()
            NPCLog("finished.",title=False,flush = True)
            NetStatistics.PrintBatchInfo(net.test_acu,-1,batch_count,prgl,net.all_test_batch_losses)
            return result


    def Test(net,dataset,compute_loss = False) -> float:

        if not hasattr(dataset,"collate_func"):
            dataset.collate_func = None
        device = ControlPanel.device
        loader = DataLoader(dataset, batch_size = ControlPanel.batch_size, shuffle=False,collate_fn=dataset.collate_func)
        prgl = math.ceil(len(dataset)/ControlPanel.batch_size)
        net.to(ControlPanel.device)
        net.eval()

        net.all_test_batch_losses = [0]
        if compute_loss:
            loss_func = net.lossf

        NPCLog("testing... ",end = "")
        net.test_acu = {True:0,False:0}
        
        batch_count,lastprg = 0,0
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            o = net(x)

            cmp_res = net.compare_output_label(o,y)
            net.test_acu[True] += torch.sum(cmp_res == True).item()
            net.test_acu[False] += torch.sum(cmp_res == False).item()

            if compute_loss:
                net.all_test_batch_losses.append(loss_func(o,y).item())
            batch_count += 1

            prgs = 100*batch_count/prgl
            if (prgs - lastprg) >= 9.7:
                NPCLog("%2.1f%%. "%prgs,end=" ",title=False,flush = True)
                lastprg = prgs

            x,y = x.cpu(),y.cpu()
        NPCLog("finished.",NetStatistics.get_acuv(net.test_acu),title=False,flush = True)

        return NetStatistics.PrintBatchInfo(net.test_acu,-1,batch_count,prgl,net.all_test_batch_losses)

    def get_parameters(net):
        return net.parameters()

    def __initTrain__(net,dataset,testdataset,lr,custom_lossf = None,optimizer = None):

        if type(dataset) != type([0]) and not hasattr(dataset,"collate_func"):
            dataset.collate_func = None

        NPCLog("net name is:",net.Name)
        NPCLog(title="")
        NPCLog("Your device is ",ControlPanel.device)
        net.to(ControlPanel.device)

        if not hasattr(net,"bestACU") and testdataset != None:
            net.bestACU = net.Test(testdataset)

        if custom_lossf == None:
            lossf = net.lossf
        else:
            lossf = custom_lossf

        if optimizer == None:
            optimizer = net.opt(net.get_parameters(), lr=lr )
        
        return optimizer,lossf,math.ceil(len(dataset)/ControlPanel.batch_size)

    @staticmethod
    def examine_data_item(item):
        x = item[0]

        return type(x) == torch.Tensor

    def __TrainNet__(net,loader_item,optimizer,lossf,device):
        """
            return batch_output,true_label,batch_loss
        """
        optimizer.zero_grad()
        x,y = loader_item
        x,y = x.to(device),y.to(device)

        o = net(x)
        loss = lossf(o,y)
        
        loss.backward()
        optimizer.step()

        return o,y,loss

    def __TrainEcho__(net,loader,optimizer,lossf,batch_echo,echo_count,prgl):
        all_batch_losses = []
        device = ControlPanel.device
        for batch_count,item in enumerate(loader):
            # if not NPCModule.examine_data_item(item):
            #     continue
            o,y,loss = net.__TrainNet__(item,optimizer,lossf,device)
            
            cmp_res = net.compare_output_label(o,y)
            net.batch_acu_statistics[True] += torch.sum(cmp_res == True).item()
            net.batch_acu_statistics[False] += torch.sum(cmp_res == False).item()
            all_batch_losses.append(torch.mean(loss).item())

            batch_count += 1

            if batch_count % batch_echo == 0:
                NetStatistics.PrintBatchInfo(net.batch_acu_statistics,echo_count,batch_count,prgl,all_batch_losses)
                net.batch_acu_statistics[True],net.batch_acu_statistics[False] = 0,0

        return all_batch_losses

    def Train(
        net,
        dataset,testdataset = None,echo = 20,
        lr = 1e-3,endACU = 99.5 , batch_echo = 10000 ,
        save = True,save_acu_lower_bound = 60 , use_train_acuv = False , 
        early_stop = True , lossf = None , loader = None , optimizer = None):

        optimizer,lossf,prgl = net.__initTrain__(dataset = dataset,testdataset = testdataset,lr = lr,custom_lossf=lossf,optimizer = optimizer)
        if loader == None:
            loader = DataLoader(dataset, batch_size = ControlPanel.batch_size, shuffle=True,collate_fn=dataset.collate_func)
        net.train_acuv_history = []
        net.train_mean_loss,net.batch_acu_statistics,saved_flag = [],{True:0,False:0},False

        net.train()
        for echo_count in range(echo):
            all_batch_losses = net.__TrainEcho__(
                loader,optimizer,lossf,
                batch_echo,echo_count,prgl
            )

            test_acuv = -1
            #statist
            net.train_mean_loss.append(mean(all_batch_losses))
            if (net.batch_acu_statistics[True]+net.batch_acu_statistics[False]) != 0:
                train_acuv = NetStatistics.PrintBatchInfo(net.batch_acu_statistics,echo_count,prgl,prgl,all_batch_losses)
                net.batch_acu_statistics = {True:0,False:0}

            ###test or eval net
            if testdataset != None:
                NPCLine()
                test_acuv = net.Test(testdataset,False)
                NPCLine()

            ###save the best net
            if test_acuv > save_acu_lower_bound and test_acuv > net.bestACU and save:
                NPCLog("There exists the best net of test acu...saved.")
                net.bestACU = test_acuv
                savename = net.save()
                saved_flag = True

            ##early stop by acuv
            if early_stop and (mean(all_batch_losses) < 1e-6 or test_acuv > endACU) and saved_flag:
                NPCLog("acu statified setting of user,program stop training the model.")
                break
        
            if early_stop and use_train_acuv and train_acuv > endACU:
                NPCLog("use last batch acuv of train.")
                NPCLog("acu ",train_acuv," statified setting of user,program stop training the model.")
                break
        
            ##early stop by history
            if use_train_acuv and testdataset == None:
                net.train_acuv_history.append(train_acuv)
            else:
                net.train_acuv_history.append(test_acuv)
            
            if early_stop and  len(net.train_acuv_history) > 4 and abs(net.train_acuv_history[-4] - net.train_acuv_history[-1]) <= 1 :
                NPCLog("trapped in a local optimum.")
                NPCLog("acu ",train_acuv," statified setting of user,program stop training the model.")
                break

        NPCLog(title="")
        if saved_flag:
            return utils.CheckModel(savename)
        elif save:
            return utils.CheckModel(net.Name,lambda:net)

        return net