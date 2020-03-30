import torch
import os
import random
from torch.utils.data import Dataset

class IMDBdataset(Dataset):
    def __init__(self, data_path, word2vec):
        self.review = []
        maindir, subdir, file_name_list = os.walk(data_path + 'pos/')
        for filename in file_name_list:
            with open(filename, 'r') as f:
                self.review.append([f.read(), 1])
            
        main_dir, sub_dir, file_name_list = os.walk(data_path + 'neg/')
        for filename in file_name_list:
            with open(filename, 'r') as f:
                self.review.append([f.read(), 0])
        random.shuffle(self.review)
        self.word2vec = word2vec
    
    def sentence2vec(self, sentence):
        input = []
        for word in sentence:
            input.append(self.word2vec[word])
        return torch.tensor(input)
    
    def __getitem__(self, index):
        return self.sentence2vec(self.review[index][0]), self.review[index][1]

    def __len__(self):
        return len(self.review)
