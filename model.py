import torch
import torch.nn as nn
import sys

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, nlayer):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, nlayer)
        self.nlayer = nlayer
        self.h0 = nn.Parameter(torch.FloatTensor(nlayer, hidden_size))
        self.c0 = nn.Parameter(torch.FloatTensor(nlayer, hidden_size))
        
        for weight in self.parameters():
            print(weight.shape)
            nn.init.normal_(weight)

    
    def forward(self, sentence):
        batch_size, lens, emb_d = sentence.shape
        print(sentence.shape)
        sentence = sentence.permute(1, 0, 2)
        h0 = torch.stack([self.h0] * batch_size, 0).permute(1, 0, 2)
        c0 = torch.stack([self.c0] * batch_size, 0).permute(1, 0, 2)
        print("Shape of c0", c0.shape)
        # c0 = torch.ones(self.nlayer, batch_size, self.hidden_size)
        output, (hn, cn) = self.lstm(sentence, (h0, c0))
        output = torch.mean(output, 0)
        return output
        
