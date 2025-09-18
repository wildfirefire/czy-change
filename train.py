import os
import six
import torch
import random
import sys
import argparse
import pickle
import numpy as np
from utils import helper
from shutil import copyfile
from sklearn import metrics
from loader import DataLoader
from trainer import GCNTrainer

# fitlog log logs
# ====== EarlyStopping 类 ======
class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.0, path="./saved_models/best_model.pt"):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self, val_metric, model):
            if self.best_score is None:
                self.best_score = val_metric
                self.save_checkpoint(model)
            elif (self.mode == "min" and val_metric > self.best_score - self.delta) or \
                (self.mode == "max" and val_metric < self.best_score + self.delta):
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_metric
                self.save_checkpoint(model)
                self.counter = 0


    def save_checkpoint(self, model):
        torch.save({'model': model.state_dict()}, self.path)
        print(f"Validation metric improved. Model saved to {self.path}")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Laptops')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')#位置
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')#词性
parser.add_argument('--dep_dim',type=int,default=30,help='Deprel embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=300, help='GCN mem dim.')
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.4, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False)
parser.add_argument('--loop', default=True)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')#0.001
parser.add_argument('--l2reg', type=float, default=1e-4, help='l2 .')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
# parser.add_argument('--seed', type=int, default=random.randint(0, 2**32 - 1))
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--beta', default=1.0e-04, type=float)
parser.add_argument('--theta', default=1.0, type=float)
parser.add_argument('--head_num', default=3, type=int, help='head_num must be a multiple of 3')
parser.add_argument('--top_k', default=2, type=int)
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optimizer', type=str, default='Adma', help='Adma; SGD')
parser.add_argument('--second_layer', type=str, default='max')
parser.add_argument('--DEVICE', type=int, default=0, help='GPU number')
parser.add_argument('--lr_scheduler', type=str, default='fixed', choices=['fixed', 'cosine', 'plateau'])
# set device

args = parser.parse_args()
args.device = torch.device("cuda:{}".format(args.DEVICE) if torch.cuda.is_available() else "cpu")

# if you want to reproduce the result, fix the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
helper.print_arguments(args)

# load contants
dicts = eval(open('./dataset/'+args.dataset+'/constant.py', 'r').read())
vocab_file = './dataset/'+args.dataset+'/vocab.pkl'
token_vocab = dict()
with open(vocab_file, 'rb') as infile:
    token_vocab['i2w'] = pickle.load(infile)
    token_vocab['w2i'] = {token_vocab['i2w'][i]:i for i in range(len(token_vocab['i2w']))}

emb_file = './dataset/'+args.dataset+'/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(token_vocab['i2w'])
assert emb_matrix.shape[1] == args.emb_dim

args.token_vocab_size = len(token_vocab['i2w'])
args.post_vocab_size = len(dicts['post'])
args.pos_vocab_size = len(dicts['pos'])
args.dep_vocab_size = len(dicts['dep'])

dicts['token'] = token_vocab['w2i']

# load training set and test set
print("Loading data from {} with batch size {}...".format(args.dataset, args.batch_size))
# train_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/train.json', args.batch_size, args, dicts)]
# test_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/test.json', args.batch_size, args, dicts)]
#restaurant
# train_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/train.json','./dataset/'+args.dataset+'/restaurant_train.raw.graph_sdat', args.batch_size, args, dicts)]
# test_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/test.json','./dataset/'+args.dataset+'/restaurant_test.raw.graph_sdat', args.batch_size, args, dicts)]

#laptops
train_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/train.json','./dataset/'+args.dataset+'/laptop_train.raw.graph_sdat', args.batch_size, args, dicts)]
test_batch = [batch for batch in DataLoader('./dataset/'+args.dataset+'/test.json','./dataset/'+args.dataset+'/laptop_test.raw.graph_sdat', args.batch_size, args, dicts)]

# create the folder for saving the best models and log file
model_save_dir = args.save_dir
helper.ensure_dir(model_save_dir, verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + args.log, header="#poch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttest_f1")

args.train_batch = train_batch
# 收集训练集标签，用于计算类别权重
train_labels = []
for batch in train_batch:
    labels = batch[-1]
    train_labels.extend(labels.cpu().numpy().tolist())

# trainer = GCNTrainer(args, emb_matrix=emb_matrix, train_labels=train_labels)
trainer = GCNTrainer(args, emb_matrix=emb_matrix)
early_stopping = EarlyStopping(patience=20, mode="max", path=args.save_dir+'/best_model.pt')

# ====== scheduler 配置 ======
if args.lr_scheduler == 'cosine':
    total_steps = len(train_batch) * args.num_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=total_steps, eta_min=1e-6)
elif args.lr_scheduler == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
else:
    scheduler = None # fixed
# ################################
train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
test_acc_history = [0.]
for epoch in range(1, args.num_epoch+1):
    print('\nepoch:%d' %epoch)
    train_loss, train_acc, train_step = 0., 0., 0
    for batch in train_batch:
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:

            print("train_loss: {:1.4f}, train_acc: {:1.4f}".format(train_loss/train_step, train_acc/train_step))

    # eval on test
    print("Evaluating on test set...")
    predictions, labels = [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for batch in test_batch:
        loss, acc, pred, label, _, _ = trainer.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        test_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("trian_loss: {:1.4f}, test_loss: {:1.4f}, train_acc: {:1.4f}, test_acc: {:1.4f}, "
          "f1_score: {:1.4f}".format(
        train_loss/train_step, test_loss/test_step,
        train_acc/train_step, test_acc/test_step,
        f1_score))
    # scheduler step
    if scheduler is not None:
        if args.lr_scheduler == 'plateau':
            scheduler.step(f1_score)
        else:
            scheduler.step()
    # 早停监控 F1
    early_stopping(f1_score, trainer.model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
        epoch, train_loss/train_step, test_loss/test_step,
        train_acc/train_step, test_acc/test_step,
        f1_score))

    train_acc_history.append(train_acc/train_step)
    train_loss_history.append(train_loss/train_step)
    test_loss_history.append(test_loss/test_step)

    # save best model
    if epoch == 1 or test_acc/test_step > max(test_acc_history):
        trainer.save(model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"
            .format(epoch, train_loss/train_step, test_loss/test_step,
            train_acc/train_step, test_acc/test_step,
            f1_score))

    test_acc_history.append(test_acc/test_step)
    f1_score_history.append(f1_score)

print("Training ended with {} epochs.".format(epoch))
bt_test_acc = max(test_acc_history)
bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
print("best test_acc/f1_score: {}/{}".format(bt_test_acc, bt_f1_score))

