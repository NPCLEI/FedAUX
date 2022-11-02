import json
import pickle
from ControlPanel import ControlPanel
from os import path
from NPCLogger import NPCBlank, NPCLog

def CheckModel(mode_name,first_train_func = None,continue_train = None,retrain = False,saveModel = True,modelfileExtendName = "pickle",pikle_root_path = ControlPanel.pickle_path):
    f_path = "%s/%s.%s"%(pikle_root_path,mode_name,modelfileExtendName)
    model = None
    if path.exists(f_path) and not retrain:
        NPCLog(" model:%s ,file exists,loading"%(mode_name),end = '')
        with open(f_path,"rb+") as model:
            model = pickle.load(model)
            NPCLog("... loaded",title="")
            if continue_train != None:
                NPCLog(" model:%s ,file exists,user choose to continue train the model"%(mode_name))
                continue_train(model)
                if saveModel:
                    SaveObj(model,mode_name)
    else:
        if retrain:
            NPCLog(" model:%s , %s ,exists,but user choose to re-train model,RETRAIN"%(f_path,mode_name),end='')
        else:
            NPCLog(" model:%s , %s ,file not exists,try to train the model"%(f_path,mode_name),end='')
        if first_train_func == None:
            NPCLog(title="")
            NPCLog(title="")
            NPCLog("EORR!! File:,",mode_name,"not exists,user donot give a able train function.AUTO SKIP!")
            NPCLog("EORR!! File:,",mode_name,"not exists,user donot give a able train function.AUTO SKIP!")
            NPCLog("EORR!! File:,",mode_name,"not exists,user donot give a able train function.AUTO SKIP!")
            NPCLog(title="")
            return None
        model = first_train_func()
        NPCLog(" trained.",title=False,end='')
        if saveModel:
            SaveObj(model,mode_name,pikle_root_path)
            NPCLog("& saved.",title=False,end='')
        NPCBlank()
    return model


def SaveObj(obj,name = "obj",path = ControlPanel.pickle_path):
    import pickle
    f_name = "%s/%s.pickle"%(path,name)
    with open(f_name, 'wb+') as net_file:
        pickle.dump(obj,net_file)

def SaveJson(obj,name = "obj",path = ControlPanel.pickle_path):
    f_name = "%s/%s.json"%(path,name)
    with open(f_name, 'w+') as js_file:
        json.dump(obj,js_file)

def LoadJson(name,path = ControlPanel.pickle_path):
    f_name = "%s/%s.json"%(path,name)
    with open(f_name, 'r') as js_file:
        return json.load(js_file)

def Counter(lst,res = {True:0,False:0}):
    for l in lst:
        if l in res:
            res[l] += 1
        else:
            res[l] = 1
    return res

CIFAR100_PATH =  r"E:/Dataset/ImgCls/CIFAR100"
CIFAR10_PATH =  r"E:/Dataset/ImgCls/CIFAR10"
EMNIST =  r"E:/Dataset/ImgCls/EMNIST"
if __name__ == "__main__":
    NPCLog(Counter([1,1,1,1,2,3,3,1]))