
import torch

from ControlPanel import ControlPanel
from .utils import NPCDataset,read_txt

class BookCorpus(NPCDataset):
    def __init__(self, data_source , read_num = 5000) -> None:
        super().__init__(data_source)
        self.read(data_source,read_num)

    def read(self, data_source ,read_num , limit_length = 512 ):
        raw_dataset = self.read_txt_folder(data_source,0)
        readtimes = 400 if ControlPanel.Debug else read_num

        for idx in range(readtimes):
            if idx % 100 == 0:
                print(idx,readtimes,idx/readtimes)
            txt = read_txt(raw_dataset[idx][0])
            try:
                l,r = 2000,7000

                while len(txt) > r:
                    self.dataset.append(
                        NPCDataset.tokenize(txt[l:r])
                    )
                    l = r
                    r = r + 5000

            except:
                pass

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx],dtype=torch.int64),0

    def get_label(self, dataset_item):
        return 0