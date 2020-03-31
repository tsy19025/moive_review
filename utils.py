import torch
import os
import random
from torch.utils.data import Dataset

class IMDBdataset(Dataset):
    def __init__(self, data_path, word2vec):
        self.review = []
        file_name_list = os.listdir(data_path + 'pos/')
        tot = 0
        self.mv = 0
        for filename in file_name_list:
            tot += 1
            if tot >= 100: break
            with open(data_path + 'pos/' + filename, 'r') as f:
                s = f.read().split(' ')
                self.mv = max(self.mv, len(s))
                self.review.append([s, 1])
            
        file_name_list = os.listdir(data_path + 'neg/')
        for filename in file_name_list:
            tot += 1
            if tot >= 200: break
            with open(data_path + 'neg/' + filename, 'r') as f:
                s = f.read().split(' ')
                self.mv = max(self.mv, len(s))
                self.review.append([s, 0])
        random.shuffle(self.review)
        self.word2vec = word2vec
    
    def sentence2vec(self, sentence):
        input = []
        for word in sentence:
            try: input.append(self.word2vec[word])
            except: continue
        data = torch.tensor(input)
        tmp = torch.zeros(self.mv - data.shape[0], data.shape[1])
        data = torch.cat([data, tmp], 0)
        return data
    
    def __getitem__(self, index):
        return self.sentence2vec(self.review[index][0]), self.review[index][1]

    def __len__(self):
        return len(self.review)
