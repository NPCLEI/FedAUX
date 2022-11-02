
import torch
from ControlPanel import ControlPanel

from NPCLogger import NPCLog
from .utils import NPCDataset

class AGNewsCsv(NPCDataset):
    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.read(data_source)

    def __getitem__(self, idx):
        res,label = self.dataset[idx]

        return (torch.tensor(res,dtype=torch.int64),torch.tensor(label - 1,dtype=torch.int64))

    def read(self, data_source , limited_length = 512):
        lines = open(data_source,'r').readlines()
        # read_times = 20 if ControlPanel.Debug else len(lines)
        read_times = len(lines)
        for idx in range(read_times):
            line = lines[idx]
            try:
                label,title,content = line.split('","')
                label = int(label[1:])
                content = title + content[:-1]
                self.register_label_idx_map(label)
                tokens = NPCDataset.tokenize(content)
                res = [NPCDataset.cls_id] + tokens + [NPCDataset.sep_id]
                ls = len(res)
                if ls < limited_length:
                    res += [0] * (limited_length - ls)
                res = res[:limited_length]
                # self.dataset.append((torch.tensor(res,dtype=torch.int64),torch.tensor(label - 1,dtype=torch.int64)))
                self.dataset.append((res,label))
            except:
                continue
