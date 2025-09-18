# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from collections import Counter
# from gcn import GCNClassifier
#
#
# class GCNTrainer(object):
#     def __init__(self, args, emb_matrix=None, train_labels=None):
#         self.args = args
#         self.emb_matrix = emb_matrix
#         self.model = GCNClassifier(args, emb_matrix=emb_matrix).cuda()
#         self.parameters = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr, weight_decay=args.l2reg)
#
#         # ==== 新增: 计算类别权重 ====
#         if train_labels is not None:
#             counts = Counter(train_labels)
#             total = sum(counts.values())
#             weights = [total / counts[i] if counts[i] > 0 else 0.0 for i in range(args.num_class)]
#             class_weights = torch.FloatTensor(weights).cuda()
#             print("类别权重:", class_weights)
#         else:
#             class_weights = None
#
#         # 使用带 class_weights 的 LabelSmoothingLoss
#         self.Loss = LabelSmoothingLoss(args.num_class, epsilon=0.1, class_weights=class_weights)
#
#         # 学习率调度器: Cosine Annealing + Warmup
#         def lr_lambda(current_step):
#             warmup_steps = 500
#             total_steps = args.num_epoch * (len(args.train_batch))
#             if current_step < warmup_steps:
#                 return float(current_step) / float(max(1, warmup_steps))
#             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#             return 0.5 * (1.0 + math.cos(math.pi * progress))
#
#         # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
#         # self.global_step = 0
#
#     # load model
#     def load(self, filename):
#         try:
#             checkpoint = torch.load(filename)
#         except BaseException:
#             print("Cannot load model from {}".format(filename))
#             exit()
#         self.model.load_state_dict(checkpoint['model'])
#         self.args = checkpoint['config']
#
#     # save model
#     def save(self, filename):
#         params = {
#             'model': self.model.state_dict(),
#             'config': self.args,
#         }
#         try:
#             torch.save(params, filename)
#             print("model saved to {}".format(filename))
#         except BaseException:
#             print("[Warning: Saving failed... continuing anyway.]")
#
#     def different_loss(self, Z, ZC):
#         diff_loss = torch.mean(torch.matmul(Z.permute(0, 2, 1), ZC) ** 2)
#         return diff_loss
#
#     def similarity_loss(self, ZCSY, ZCSE):
#         ZCSY = F.normalize(ZCSY, p=2, dim=1)
#         ZCSE = F.normalize(ZCSE, p=2, dim=1)
#         similar_loss = torch.mean((ZCSY - ZCSE) ** 2)
#         return similar_loss
#
#     def update(self, batch):
#         inputs = batch[0:12]
#         label = batch[-1]
#
#         # step forward
#         self.model.train()
#         self.optimizer.zero_grad()
#         logits, gcn_outputs, h_sy, h_se, h_csy, h_cse = self.model(inputs)
#
#         diff_loss = 0
#         similar_loss = 0
#         loss = self.Loss(logits, label) + diff_loss + similar_loss
#         corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
#         acc = 100.0 * float(corrects) / label.size()[0]
#
#         # backward
#         loss.backward()
#         self.optimizer.step()
#         return loss.data, acc
#
#     def predict(self, batch):
#         inputs = batch[0:12]
#         label = batch[-1]
#
#         # forward
#         self.model.eval()
#         logits, gcn_outputs, h_sy, h_se, h_csy, h_cse = self.model(inputs)
#
#         diff_loss = 0
#         similar_loss = 0
#         loss = self.Loss(logits, label) + diff_loss + similar_loss
#         corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
#         acc = 100.0 * float(corrects) / label.size()[0]
#         predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
#         predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
#
#         return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, gcn_outputs.data.cpu().numpy()
#
#
# # =====================
# # 平滑交叉熵 + 类别权重
# # =====================
# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, num_classes, epsilon=0.1, class_weights=None):
#         super().__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.class_weights = class_weights  # tensor: [num_classes] or None
#
#     def forward(self, logits, labels):
#         log_probs = F.log_softmax(logits, dim=-1)
#         one_hot = F.one_hot(labels, self.num_classes).float()
#         smooth_label = (1 - self.epsilon) * one_hot + self.epsilon / self.num_classes
#
#         per_sample_loss = -torch.sum(smooth_label * log_probs, dim=-1)
#
#         # # 如果有 class_weights，则按类别加权
#         # if self.class_weights is not None:
#         #     weights = self.class_weights[labels]
#         #     per_sample_loss = per_sample_loss * weights
#
#         return per_sample_loss.mean()



import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn import GCNClassifier


class GCNTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(args, emb_matrix=emb_matrix).cuda()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr,weight_decay=args.l2reg)
        self.Loss = LabelSmoothingLoss(args.num_class)

        # 学习率调度器: Cosine Annealing + Warmup
        def lr_lambda(current_step):
            warmup_steps = 500

            total_steps = args.num_epoch * (len(args.train_batch))
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # self.global_step = 0
    # load model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def different_loss(self, Z, ZC):
        diff_loss = torch.mean(torch.matmul(Z.permute(0, 2, 1), ZC) ** 2)
        return diff_loss

    def similarity_loss(self, ZCSY, ZCSE):
        ZCSY = F.normalize(ZCSY, p=2, dim=1)
        ZCSE = F.normalize(ZCSE, p=2, dim=1)
        similar_loss = torch.mean((ZCSY - ZCSE) ** 2)
        return similar_loss

    def update(self, batch):
        inputs = batch[0:12]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs, h_sy, h_se, h_csy, h_cse= self.model(inputs)

        # diff_loss = self.args.beta * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        # similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)
        diff_loss=0
        similar_loss=0
        loss = self.Loss(logits, label) + diff_loss + similar_loss
        # loss = F.cross_entropy(logits, label, reduction='mean') + diff_loss + similar_loss
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()

        acc = 100.0 * float(corrects) / label.size()[0]

        # backward
        loss.backward()
        self.optimizer.step()
        #余弦退火
        # self.global_step += 1
        # self.scheduler.step()
        return loss.data, acc

    def predict(self, batch):
        inputs = batch[0:12]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs, h_sy, h_se, h_csy, h_cse = self.model(inputs)

        # diff_loss = self.args.beta * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        # similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)
        diff_loss = 0
        similar_loss = 0
        loss = self.Loss(logits, label) + diff_loss + similar_loss
        # loss = F.cross_entropy(logits, label, reduction='mean') + diff_loss + similar_loss
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        # acc = 100.0 * np.float(corrects) / label.size()[0]
        acc = 100.0 * float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, gcn_outputs.data.cpu().numpy()
#平滑正则化
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon  # 平滑系数

    def forward(self, logits, labels):
        # logits: (batch, num_classes), labels: (batch,)
        log_probs = F.log_softmax(logits, dim=-1)
        # 计算平滑标签：(1-ε)*one_hot + ε/num_classes
        one_hot = F.one_hot(labels, self.num_classes).float()
        smooth_label = (1 - self.epsilon) * one_hot + self.epsilon / self.num_classes
        # 交叉熵损失
        loss = -torch.sum(smooth_label * log_probs, dim=-1).mean()
        return loss