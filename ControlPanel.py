
import json
import torch
import platform
import NPCLogger
from pytorch_transformers import BertConfig

# cpu_num是一个整数
torch.set_num_threads(16)


class ControlPanel:
    @staticmethod
    def get_files_path(file_name):
        if ControlPanel.isWindows():
            return ControlPanel._windows_model_files_paths[file_name]
        else:
            return ControlPanel._linux_model_files_paths[file_name]

    @property
    def model_files_paths():
        if ControlPanel.isWindows():
            return ControlPanel._windows_model_files_paths
        else:
            return ControlPanel._linux_model_files_paths
    
    _linux_model_files_paths = {
        "tiny_bert":"/root/autodl-tmp/FedPick/Models/TinyModules/TinyBert",
        "bert_config_path":"/root/autodl-tmp/FedPick/Configs/config.json"
    }

    _windows_model_files_paths = {
        "tiny_bert":"./Models/TinyModules/TinyBert",
        "bert_config":"./Configs/config.json"
    }

    cur_ex_cls_num = 2

    Debug = False
    """
        Debug = True 只考虑程序能否顺利运行,不考虑模型训练情况
    """

    CurExName = "FedLearning"

    max_commu_times = 50

    read_data_num = 100

    pickle_path = "F:/Pickle"

    batch_size = 32

    global_lr = 1e-4
    
    envir_path = "./"
    
    data_path = "/root/autodl-tmp/Dataset"
    
    _bert_config_path_ = None

    def WorkSpace(name):
        import os
        ControlPanel.envir_path = "%s/%s"%(ControlPanel.pickle_path,name)
        if not os.path.exists(ControlPanel.envir_path):
            os.mkdir(ControlPanel.envir_path)
            os.mkdir(ControlPanel.envir_path + '/Log/')

        if not NPCLogger.inited:
            NPCLogger.initLogger(ControlPanel.envir_path)

            NPCLogger.NPCLog(" your system is ",ControlPanel.envir_system)

        ControlPanel.CurExName = name
        ControlPanel.pickle_path = ControlPanel.envir_path
        # NPCLogger.begain_write_num_line = 20

    @property
    def bert_config():
        if ControlPanel.isWindows():
            return 

        BertConfig.from_json_file(ControlPanel._linux_model_files_paths['bert_config'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envir_system = platform.system()

    bertTokenizer = None

    def isWindows():
        return ControlPanel.envir_system == "Windows"

    @staticmethod
    def init():
        if ControlPanel.envir_system != "Windows":
            ControlPanel.bert_config = BertConfig.from_json_file("/root/autodl-tmp/FedPick/Configs/config.json")
            ControlPanel.envir_path = "/root/autodl-tmp/FedPick"
            ControlPanel.pickle_path = "/root/autodl-tmp/ModelPickle"
            ControlPanel.data_path = "/root/autodl-tmp/Dataset"
            ControlPanel.batch_size = 32
            
        # if not NPCLogger.inited:
        #     NPCLogger.initLogger(ControlPanel.envir_path)

        #     NPCLogger.NPCLog(" your system is ",ControlPanel.envir_system)

    @staticmethod
    def CPU():
        """
            强制接下来的模型进入CPU训练(不能中断正在训练的模型)
        """
        ControlPanel.device = torch.device("cpu")

    @staticmethod
    def Tokenizer(packageName = 'pytorch_transformers'):

        if ControlPanel.bertTokenizer != None:
            return ControlPanel.bertTokenizer
        if packageName == 'pytorch_transformers':
            from pytorch_transformers import BertTokenizer
        else:
            from transformers import BertTokenizer
        ControlPanel.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return ControlPanel.bertTokenizer

    @staticmethod
    def GetBERT(packageName = 'pytorch_transformers'):

        if packageName == 'pytorch_transformers':
            from pytorch_transformers import BertModel
        else:
            from transformers import BertModel
        
        return BertModel.from_pretrained('bert-base-uncased')

    @property
    def tokenizer():
        return ControlPanel.Tokenizer()

ControlPanel.init()