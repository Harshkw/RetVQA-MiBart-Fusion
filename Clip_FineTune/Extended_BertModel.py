import torch
import torch.nn as nn

class BertWithLinear(nn.Module):
    def __init__(self, bert_model):
        super(BertWithLinear, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(768, 512)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        sequence_output = outputs.last_hidden_state.mean(dim=1)
        reduced_sequence_output = self.linear(sequence_output)
        return sequence_output, reduced_sequence_output