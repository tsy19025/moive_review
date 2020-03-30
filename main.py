import torch
import torch.nn as .nn
from gensim.models import Word2Vec
import sys
import os
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type = int, default = 100)
    parse.add_argument('--batch_size', type = int, default = 64)
    parse.add_argument('--word2vec_save', type = str, default = "word2vec")
    parse.add_argument('--word2vec', type = bool, default = True)
    parse.add_argument('--save', type = str, default = "model/model.pth")
    return parse.parse_args()

def get_word2vec(args):
    sentences = ""
    for data_path in ['train', 'valid', 'test']:
        maindir, subdir, file_name_list = os.walk(data_path + 'pos/')
        for filename in file_name_list:
            with open(filename, 'r') as f:
                sentences = sentences + f.read() + '\n'
        maindir, subdir, file_name_list = os.walk(data_path + 'neg/')
        for filename in file_name_list:
            with open(filename, 'r') as f:
                sentences = sentences + f.read() + '\n'
    with open('./all_sentences.txt', 'w') as f2:
        f2.write(sentences)
    
    sentences = word2vec.LineSentence('all_sentences.txt')
    word2vec = Word2Vec(sentences, hs = 1, min_count = 5, window = 5,size = 512)
    word2vec.save(args.word2vec_save)

def train_one_epoch(model, train_data_loader, optimizer, loss_fn, device):
    train_loss = []
    for step, data in enumerate(train_data_loader):
        input, label = data
        input = input.to(device)
        label = label.to(device)
        
        output = model(input)
        loss = loss_fn(output, label)

def same(a, b):
    tot = 0
    for i in range(a):
        if b[i] >= 0.5 and a[i] == 1: tot += 1
        elif b[i] < 0.5 and a[i] == 0: tot += 1
    return tot

def valid(model, valid_data_loader):
    current = 0
    tot = 0
    for step, data in enumerate(train_data_loader):
        input, label = data
        input = input.to(device)
        label = label.to(device)
        
        output = model(input)
        tot += len(output)
        current += same(label, output)
    return 1.0 * current / tot
  
if "__name__" == __main__:
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    
    if args.word2vec == True:
        get_word2vec(args, args.data_path)
    
    word2vec = Word2Vec.load(args.word2vec_save)  
    train_data_loader = DataLoader(dataset = IMDBdataset('train/', word2vec),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20, pin_memory = True)
    valid_data_loader = DataLoader(dataset = IMDBdataset('valid/', word2vec),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20, pin_memory = True))
    test_data_loader = DataLoader(dataset = IMDBdataset('test/', word2vec),
                                       batch_size = args.batch_size,
                                       shuffle = True,
                                       num_workers = 20, pin_memory = True))
    
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = Model()
    model = model.to(device)
    
    loss_fn = nn.BCELoss(reduction = 'none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
                                 weight_decay = 0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    
    best_acc = 0
    best_epoch = -1
    for epoch in args.epochs:
        print("Epoch", epoch);
        mean_loss = train_one_epoch(model, train_data_loader, optimizer, 
                                    loss_fn, epoch)
        acc = valid(model, valid_data_loader)
        print("Valid: Acc", acc)
        if acc > best_acc:
            acc = best_acc
            best_epoch = epoch
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, args.save)
            print('Model save for better valid acc:', best_acc)
        elif epoch - best_epoch > 5:
            print("Steop training at epoch", epoch)
            break
    # Test
    state = torch.load(args.save)
    model.load_state_dict(state['net'])
    model.to(device)
    acc = valid(test_data_load, test_data_loader)
    print("Test: Acc", acc)
