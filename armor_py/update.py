import numpy as np
from sklearn import metrics
from torch import autograd
from torch.utils.data import DataLoader, Dataset
import copy
from attack.projected_gradient_descent import projected_gradient_descent
import torch
import torch.nn as nn


def armor_loss(global_out_adv, local_out, global_out_clean, eta, labels, target):
    def cos_loss(global_out_adv, local_out):
        cos_sim = nn.CosineEmbeddingLoss()
        cosine_loss = eta * cos_sim(global_out_adv, local_out, target)
        return cosine_loss

    def cross_entropy(global_out_adv, global_out_clean, labels):
        loss_fn_cross = nn.CrossEntropyLoss()
        loss = loss_fn_cross(global_out_adv, labels) + loss_fn_cross(global_out_clean, labels)
        return loss

    ce_loss = cross_entropy(global_out_adv, global_out_clean, labels)
    diff_loss = cos_loss(global_out_adv, local_out)
    return ce_loss + diff_loss, ce_loss, diff_loss


class Similarity_AT(object):
    def __init__(self, args, dataset_test, at_idxs, test_idxs, eta, epoch, device):
        self.args = args
        self.dataset_test = dataset_test
        self.at_idxs = at_idxs
        self.test_idxs = test_idxs
        self.batch_size = args.batch_size
        self.epoch = epoch
        self.device = device
        self.eta = eta

    def at_update_weights(self, global_model, local_model, p, chosen_at, chosen_test, eps, eps_step, iter_round):
        global_net = copy.deepcopy(global_model)
        global_net.train()
        local_net = copy.deepcopy(local_model)
        local_net.eval()
        epoch_loss, epoch_ce_loss, epoch_diff_loss = [], [], []
        epoch_acc = []
        optimizer = torch.optim.SGD(global_net.parameters(), lr=self.args.lr, momentum=0)
        at_dataset = DataLoader(DatasetSplit(self.dataset_test, chosen_at), batch_size=self.batch_size, shuffle=True)
        test_acc, test_loss = 0.0, 0.0
        batch_acc, batch_loss, batch_ce_loss, batch_diff_loss = [], [], [], []
        for iter in range(self.epoch):
            for images, labels in at_dataset:
                global_net.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                images_pgd = projected_gradient_descent(copy.deepcopy(global_net), images, eps, eps_step,
                                                        iter_round, np.inf)
                optimizer.zero_grad()
                log_probs = global_net(images)
                y_pred = np.argmax(log_probs.cuda().data.cpu(), axis=1)
                acc = metrics.accuracy_score(y_true=labels.cuda().data.cpu(), y_pred=y_pred)
                target = torch.bernoulli(p * torch.ones(log_probs.shape[0])).to(self.device)
                loss, ce_loss, diff_loss = armor_loss(global_net(images_pgd), local_net(images_pgd), global_net(images), self.eta, labels, target)
                loss.backward()
                optimizer.step()
                batch_acc.append(acc)
                batch_loss.append(loss.data.item())
                batch_ce_loss.append(ce_loss.data.item())
                batch_diff_loss.append(diff_loss.data.item())
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_ce_loss.append(sum(batch_ce_loss) / len(batch_ce_loss))
            epoch_diff_loss.append(sum(batch_diff_loss) / len(batch_diff_loss))
            test_acc, test_loss = self.at_test(copy.deepcopy(global_net), copy.deepcopy(local_net), p, chosen_test)
            print("train epoch", iter, ":acc={:.10f}".format(sum(batch_acc) / len(batch_acc)),
                  "test_acc={:.10f}".format(test_acc),
                  "loss={:.10f}".format(sum(batch_loss) / len(batch_loss)),
                  "ce_loss={:.10f}".format(sum(batch_ce_loss) / len(batch_ce_loss)),
                  "diff_loss={:.10f}".format(sum(batch_diff_loss) / len(batch_diff_loss)))

        avg_acc = sum(epoch_acc) / len(epoch_acc)
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_ce_loss = sum(epoch_ce_loss) / len(epoch_ce_loss)
        avg_diff_loss = sum(epoch_diff_loss) / len(epoch_diff_loss)

        print("train avg", ": loss={:.10f}".format(avg_loss), " ce_loss={:.10f}".format(avg_ce_loss),
              " diff_loss={:.10f}".format(avg_diff_loss))
        print("avg_acc={:.10f}".format(avg_acc), "final_acc={:.10f}".format(epoch_acc[self.epoch - 1]))
        if self.args.dataset == "cifar":
            threshold = 0.5
        elif self.args.dataset == "mnist":
            threshold = 0.8
        else:
            print("dataset error")

        if test_acc >= threshold:
            acc_flag = 1
        else:
            acc_flag = 0
            print("#################### Training Failed ####################")
            print("################# Resample and Restart ##################")

        w = global_net.state_dict()
        return w, epoch_loss[self.epoch - 1], epoch_acc[self.epoch - 1], test_acc, acc_flag

    def at_test(self, global_net, local_net, p, dict_test):
        log_probs = []
        labels = []
        at_test_dataset = DataLoader(DatasetSplit(self.dataset_test, dict_test), batch_size=self.batch_size,
                                     shuffle=True)
        batch_loss, batch_ce_loss, batch_diff_loss = [], [], []
        loss, ce_loss, diff_loss = 0.0, 0.0, 0.0
        for images, labels in at_test_dataset:
            images = images.to(self.device)
            labels = labels.to(self.device)
            global_net = global_net.float()
            log_probs = global_net(images)
            global_net.eval()
            target = torch.bernoulli(p * torch.ones(log_probs.shape[0])).to(self.device)
            loss, ce_loss, diff_loss = armor_loss(global_net(images), local_net(images), global_net(images), self.eta, labels, target)
            batch_loss.append(loss.data.item())
            batch_ce_loss.append(ce_loss.data.item())
            batch_diff_loss.append(diff_loss.data.item())
        # avg_loss = sum(batch_loss) / len(batch_loss)
        # avg_ce_loss = sum(batch_ce_loss) / len(batch_ce_loss)
        # avg_diff_loss = sum(batch_diff_loss) / len(batch_diff_loss)
        y_pred = np.argmax(log_probs.cuda().data.cpu(), axis=1)
        acc = metrics.accuracy_score(y_true=labels.cuda().data.cpu(), y_pred=y_pred)
        return acc, loss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(device)
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split train and test
        idxs_train = idxs
        if (self.args.dataset == 'mnist') or (self.args.dataset == 'cifar'):
            idxs_test = idxs
            train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.batch_size, shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        else:
            train = self.args.dataset_train[idxs]
            test = self.args.dataset_test[idxs]
        return train, test

    def update_weights(self, net, device):
        net.to(device)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)

        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                images = images.to(device)
                labels = labels.to(device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc, _, = self.test(net, device=device)
            epoch_acc.append(acc)
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_acc = sum(epoch_acc) / len(epoch_acc)
        w = net.state_dict()
        return w, avg_loss, avg_acc

    def test(self, net, device):
        loss = 0
        log_probs = []
        labels = []
        net.to(device)
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            images, labels = autograd.Variable(images).to(device), autograd.Variable(labels).to(device)
            images = images.to(device)
            labels = labels.to(device)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
        y_pred = np.argmax(log_probs.cuda().data.cpu(), axis=1)
        acc = metrics.accuracy_score(y_true=labels.cuda().data.cpu(), y_pred=y_pred)
        loss = loss.cuda().data.cpu().item()
        return acc, loss