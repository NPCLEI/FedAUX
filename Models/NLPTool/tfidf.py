import ControlPanel
from pytorch_transformers import BertTokenizer

def setDictItem(mp:dict,tid,increment = 1,dv = 1,mean_instead_increment = False):
    if tid in mp:
        if mean_instead_increment:
            mp[tid] = (mp[tid] + increment) // 2
        else:
            mp[tid] += increment
    else:
        mp.setdefault(tid,dv)

class TFMap:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self) -> None:
        self.table = {}
    
    def Statistics(self,txt,increment = 1,mean_instead_increment = False):
        ids = TFMap.tokenizer.encode(txt)
        for tid in ids:
            setDictItem(self.table,tid,1,1)

    def __getitem__(self,idx):
        return self.table[idx]
    
    def MergeOtherTFMap(self,other):
        for key in other.table:
            setDictItem(self.table,key,other[key],other[key])

    def deepcopy(self):
        res = TFMap()
        res.table = dict(list(self.table.items()))
        return res

class IDFMap:
    def __init__(self,tfmaps = [],merge_mode = 'mean') -> None:
        self.maintfmap = TFMap()
        self.othertfmap = TFMap()
        self.mode = merge_mode
        if len(tfmaps) != 0:
            self.maintfmap = tfmaps[0].deepcopy()
            for mpidx in range(1,len(tfmaps)):
                self.othertfmap.MergeOtherTFMap(tfmaps[mpidx])

    def __getitem__(self,idx):
        return self.maintfmap[idx]/self.othertfmap[idx]

if __name__ == "__main__":
    tfmap = TFMap()
    tfmap.Statistics("It's really impressive.")

    idfmap = IDFMap([tfmap,tfmap,tfmap,tfmap,tfmap,tfmap])
    print(idfmap.maintfmap.table)
    print(idfmap.othertfmap.table)
    print(idfmap[2009])