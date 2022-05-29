import math

import numpy as np
import torch
from easydict import EasyDict
from torchvision import datasets, transforms

from armor_py.models import CNN_MNIST, CNN_CIFAR


def create_model(args, device):
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_MNIST()
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNN_CIFAR()
    else:
        exit('Error: unrecognized model')
    net_glob.to(device)
    return net_glob


def at_sample(dataset_test, at_ratio):
    num_items = int(len(dataset_test) * at_ratio)
    dict_at, dict_test, all_idxs = {}, {}, [i for i in range(len(dataset_test))]
    at_idxs_set = set(np.random.choice(all_idxs, num_items, replace=False))
    test_idxs_set = set(all_idxs) - at_idxs_set
    at_idxs = list(at_idxs_set)
    test_idxs = list(test_idxs_set)
    return at_idxs, test_idxs


def sample_user(args, dataset_train, dataset_test):
    if args.dataset == "mnist" and args.iid == 0:
        dict_train = mnist_noniid(dataset_train, args.client_num_in_total)
        dict_test = mnist_iid(dataset_test, args.client_num_in_total)
    elif args.dataset == "cifar" and args.iid == 0:
        dict_train = cifar_noniid(dataset_train, args.client_num_in_total)
        dict_test = cifar_iid(dataset_test, args.client_num_in_total)
    else:
        exit('Error: unrecognized dataset')
    return dict_train, dict_test


def load_data(args):
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./dataset/mnist/', train=False, download=True, transform=transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=transform)
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test


def ld_cifar10(batch_size):
    test_transforms = transforms.ToTensor()
    server_dataset = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=test_transforms)
    client_dataset = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=test_transforms)
    server_loader = torch.utils.data.DataLoader(server_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    return EasyDict(server=server_loader, client=client_loader)


def ld_mnist(batch_size):
    test_transforms = transforms.ToTensor()
    server_dataset = datasets.MNIST('./dataset/mnist/', train=False, download=True, transform=test_transforms)
    client_dataset = datasets.MNIST('./dataset/mnist/', train=True, download=True, transform=test_transforms)
    server_loader = torch.utils.data.DataLoader(server_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    return EasyDict(server=server_loader, client=client_loader)


def mnist_iid(dataset, num_users):
    num_items = math.floor(dataset.data.shape[0] / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    chosen_shards = 3
    num_shards = num_users * chosen_shards
    num_imgs = math.floor(dataset.data.shape[0] / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(dataset.data.shape[0])
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    num_items = math.floor(dataset.data.shape[0] / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    chosen_shards = 4
    num_shards = num_users * chosen_shards
    num_imgs = math.floor(dataset.data.shape[0] / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(dataset.data.shape[0])
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, chosen_shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
