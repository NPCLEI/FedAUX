# print(tiny_bert_bin['bert.embeddings.LayerNorm.bias'],tiny_bert_bin['bert.embeddings.LayerNorm.bias'].size())
# for parm in tiny_bert.parameters():
#     print(parm.data == tiny_bert_bin['bert.embeddings.LayerNorm.bias'])

import torch

from numpy.random import shuffle
from ControlPanel import ControlPanel
from Dataset.utils import NPCDataset as NPCDataset
from pytorch_transformers import BertConfig,BertModel,BertForMaskedLM,BertTokenizer

def get_tiny_bert(path = ControlPanel.get_files_path('tiny_bert')) -> BertModel:


    tiny_bert_bin = torch.load('%s/tiny_bert.bin'%path)

    tiny_bert = BertModel.from_pretrained("%s/tiny_bert.json"%path,state_dict=tiny_bert_bin)

    return tiny_bert

class NextTokenPredict(NPCDataset):
    cls_id  = 101
    sep_id  = 102
    mask_id = 103

    def __init__(self ,dataset, mask_ratio = 0.15) -> None:
        super(NextTokenPredict, self).__init__()
        self.dataset = dataset
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx):
        tokens_ids = self.get_data(self.dataset[idx])
        res_ids = torch.clone(tokens_ids)
        input_ids_length = len(res_ids)
        idxes = [idx for idx in range(input_ids_length)]
        shuffle(idxes)
        mask_idxes = idxes[:int(self.mask_ratio * input_ids_length)]
        res_ids[mask_idxes] = NextTokenPredict.mask_id
        
        """
            mask_input_ids,mask_idxes,or_input_ids
        """
        return res_ids,mask_idxes,tokens_ids

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tiny_bert_bin = torch.load('./tiny_bert.bin')
# tiny_bert = BertForMaskedLM.from_pretrained("./tiny_bert.json",state_dict=tiny_bert_bin)
# input_ids = torch.tensor(tokenizer.encode("Hello, my [MASK] is cute")).unsqueeze(0)  # Batch size 1

# print('0',input_ids)
# outputs = tiny_bert(input_ids, masked_lm_labels=input_ids)
# loss, prediction_scores = outputs[:2]
# loss.backward()

# predict = torch.argmax(prediction_scores,dim=2)
# print(loss,torch.argmax(prediction_scores,dim=2))
# print(tokenizer.decode(predict[0].tolist()))

# if __name__ == "__main__":
    
