import copy
import os
import random
import re
import shutil
import torch
import numpy as np
import wandb
from armor_py.update import LocalUpdate


def local_test_on_all_clients(args, net_glob, dataset_test, dict_server, device):
    list_acc, list_loss = [], []
    for c in range(args.client_num_in_total):
        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_server[c], device=device)
        acc, loss = net_local.test(net=net_glob, device=device)
        list_acc.append(acc)
        list_loss.append(loss)
    return list_acc, list_loss


def dict_avg(dict_name):
    dict_len = len(dict_name)
    dict_sum = sum(dict_name.values())
    dict_avg = dict_sum / dict_len
    return dict_avg


def wandb_init(args):
    # offline
    # os.environ["WANDB_MODE"] = "offline"

    # online
    # input your wandb api key here
    os.environ["WANDB_API_KEY"] = "f55c26ca1afa4b1886def24a903c98b48d80253e"

    run = wandb.init(reinit=True, project="armor-" + args.dataset,
                     name="num of client=" + str(args.client_num_in_total) + ",no noise" +
                          ",model=" + str(args.model) + ",lr=" + str(args.lr) + ",round=" + str(args.comm_round),
                     config=args)
    return run


def test_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def test_cpdir(source, target):
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(source, target)


def alter(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def alter_re(file, pattern, repl):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(pattern, repl, line)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def del_blank_line(file):
    file_data = ""
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            if line.split():
                file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def fix_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


def aggregate(w):
    w_avg = copy.deepcopy(w[0])
    if isinstance(w[0], np.ndarray):
        for i in range(1, len(w)):
            w_avg += w[i]
        w_avg = w_avg / len(w)
    else:
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def del_tensor_element(tensor, index):
    top = tensor[0:index]
    tail = tensor[index + 1:]
    new_tensor = torch.cat((top, tail), dim=0)
    return new_tensor
