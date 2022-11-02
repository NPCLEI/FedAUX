from copy import deepcopy
from FedFMK.FedUtils import FedDataShell
from ControlPanel import ControlPanel
from NPCLogger import NPCLog,NPCBlank
from Models.nn import NPCModule
from NPCLogger import NPCLog,NPCBlank,NPCLogTitleContext


class Client_:
    """
        for server debug
    """
    def __init__(self) -> None:
        self.id = -1

    def SetLoader(self,initObject):
        self.DataObject = initObject

    def DownLoadModel(self,model:NPCModule):
        pass

    def TestModel(self):
        pass

    def TrainModel(self,echo = 5):
        pass

    def EndCommuc(self):
        pass

class Client:
    def __init__(self,train_dataset:FedDataShell,test_datset:FedDataShell = None,split = False,split_ration = 0.5) -> None:
        """
            if test_datset == None, Client will split part of train to test.
        """
        if test_datset != None or not split:
            self.train_dataset = train_dataset
            self.test_datset = test_datset
        else:
            self.train_dataset,self.test_datset = train_dataset.Split(split_ration)
            # self.train_dataset.PrintLabelDistribution()
            # self.test_datset.PrintLabelDistribution()

        self.id = -1
        self.model = None
        self.save_model = False
        self.cur_server_comicn = -1
    
    @property
    def LogTitleContext(self):
        return NPCLogTitleContext("client %d report"%self.id)

    def SetLoader(self,initObject):
        self.DataObject = initObject

    def DownLoadModel(self,model:NPCModule,modelShell = None):
        # self.model = NNInterFace.CloneModule(model,modelShell)
        self.model = deepcopy(model)
        self.model.clear_states()

    def TestModel(self):
        with NPCLogTitleContext("client %d report"%self.id):
            NPCLog("client report")
            return self.model.Test(self.test_datset,compute_loss=True)

    def TrainModel(self,echo = 5,test = False):
        with NPCLogTitleContext("client %d report"%self.id):
            self.model.Train(self.train_dataset,self.test_datset if test else None,echo=echo,batch_echo=10000,save=False,lr=ControlPanel.global_lr,early_stop=False)

    def EndCommuc(self):
        """
            transfer the model to memo ,save memo of gpu
        """
        if self.model == None:
            return
        self.model.cpu()

from multiprocessing import Process
import utils
class PClient:
    def __init__(self,train_dataset:FedDataShell,test_datset:FedDataShell = None,split = False,split_ration = 0.5) -> None:
        """
            if test_datset == None, Client will split part of train to test.
        """
        if test_datset != None or not split:
            self.train_dataset = train_dataset
            self.test_datset = test_datset
        else:
            self.train_dataset,self.test_datset = train_dataset.Split(split_ration)
            # self.train_dataset.PrintLabelDistribution()
            # self.test_datset.PrintLabelDistribution()

        self.id = -1
        self.model = None
        self.global_model_path = None

    def SetLoader(self,initObject):
        self.DataObject = initObject

    def DownLoadModel(self,model_path):
        # self.model = NNInterFace.CloneModule(model,modelShell)
        self.global_model_path = model_path

    def TestModel(self):
        with NPCLogTitleContext("client %d report"%self.id):
            NPCLog("client report")
            return self.model.Test(self.test_datset,compute_loss=True)

    def TrainModel(self,echo = 5,test = False):
        self.process = Process(
                target=self.TrainModelProcess,
                args=(echo,test),
            )
        self.process.start()

            
    def EndCommuc(self):
        """
            transfer the model to memo ,save memo of gpu
        """
        self.process.join()
        if self.model == None:
            return
        self.model.cpu()
