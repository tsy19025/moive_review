import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
    
    def forward(self, sentence):
        # sentences: [nword, batch_size, input_size]
        # output: 
        # padding? pack_padded_sequence
        output = self.lstm(sentence, self.h0, self.c0)
        
        output = torch.mean()
        
        
