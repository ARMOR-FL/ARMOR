import copy
import math
import os
import numpy as np
import torch
from torch import device
from armor_py.utils import fix_random
from armor_py.options import args_parser
from armor_py.sampling import load_data, create_model, at_sample
from armor_py.update import Similarity_AT


def retrain(dataset_test, at_idxs, test_idxs):
    print("at retain, dataset =", args.dataset, ", num_users =", args.client_num_in_total)
    fix_random(args.random_seed)
    percent_sub = args.percent_sub
    eta = args.eta
    p = args.p
    Client, Sub_Fed, dict_sub_idx, Client_at = {}, {}, {}, {}
    epoch = 1000
    num_sub = math.ceil(percent_sub * args.client_num_in_total)
    # make directory
    model_path = "fl_model/{}/client_num_{}/".format(args.dataset, args.client_num_in_total)
    diff_path = "at_model/{}/client_num_{}/".format(args.dataset, args.client_num_in_total)
    at_path = diff_path + "Client_sub_{:.2f}_eta_{}_epoch_{:d}_p_{:.2f}/".format(percent_sub, eta, epoch, p)
    if not os.path.exists(at_path):
        os.makedirs(at_path)
    # load model
    w_glob = torch.load(model_path + "Global.pth")
    net_glob = create_model(args, device)
    net_glob.load_state_dict(w_glob)
    final_path = model_path + "Client_last/"
    for idx in range(args.client_num_in_total):
        Client[idx] = torch.load(final_path + "Client_last_{}.pth".format(idx))
    # training
    print("###############################################################")
    print("dataset_test:", dataset_test, "Size of sub-fed:", num_sub, "percent:", percent_sub, "eta=", eta, "epoch=",
          epoch, "p=", p, "\n")
    loss_ats, acc_ats, test_acc_ats = [], [], []
    dict_at, dict_test = {}, {}
    sim_at = Similarity_AT(args=args, dataset_test=dataset_test, at_idxs=at_idxs, test_idxs=test_idxs, eta=eta,
                           epoch=epoch, device=device)
    global_acc, global_loss = sim_at.at_test(copy.deepcopy(net_glob), copy.deepcopy(net_glob), p, test_idxs)
    print("Global test: acc=", global_acc, "loss=", global_loss)
    print('Server AT begins...')
    for idx in range(args.client_num_in_total):
        acc_flag = 0
        count_retry = 0
        while acc_flag == 0:
            # generate sub-fed model
            dict_sub_idx[idx] = np.random.choice(range(args.client_num_in_total), num_sub, replace=False)
            Sub_Fed[idx] = copy.deepcopy(Client[dict_sub_idx[idx][0]])
            if num_sub > 1:
                for key, value in Client[idx].items():
                    for i in range(num_sub - 1):
                        Sub_Fed[idx][key] += Client[dict_sub_idx[idx][i + 1]][key]
                    Sub_Fed[idx][key] = Sub_Fed[idx][key] / num_sub
            dict_at[idx] = set(np.random.choice(at_idxs, args.at_num, replace=False))
            dict_test[idx] = set(np.random.choice(test_idxs, args.test_num, replace=False))
            # sub-fed test
            local_model = create_model(args, device)
            local_model.load_state_dict(Sub_Fed[idx])
            sub_acc, sub_loss = sim_at.at_test(copy.deepcopy(local_model), copy.deepcopy(local_model), p, dict_test[idx])
            print("Sub_Fed test: acc=", sub_acc, "loss=", sub_loss.item())
            print("Client {} after AT of all {} Clients:".format(idx, args.client_num_in_total))
            w, loss, acc, test_acc, acc_flag = sim_at.at_update_weights(net_glob, local_model, p,
                                                                        dict_at[idx], dict_test[idx],
                                                                        eps, eps_step, iter_round)
            count_retry += 1
            if count_retry > 0:
                print("Client {} count: {}\n".format(idx, count_retry))
        Client_at[idx] = w
        torch.save(Client_at[idx], at_path + "Client_at_{}.pth".format(idx))
        loss_ats.append(loss)
        acc_ats.append(acc)
        test_acc_ats.append(test_acc)

    loss_avg_ats = sum(loss_ats) / len(loss_ats)
    acc_avg_ats = sum(acc_ats) / len(acc_ats)
    test_acc_avg_ats = sum(test_acc_ats) / len(test_acc_ats)
    print('\nFinal AT model loss:', loss_avg_ats)
    print('\nFinal AT model acc:', acc_avg_ats)
    print('\nFinal AT model test acc:', test_acc_avg_ats)
    print("Retrain of dataset = " + args.dataset + ", num of client = {} completed!".format(args.client_num_in_total))


if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda)
    device = torch.device("cuda:0")
    pid = os.getpid()
    print('PID No.', pid)
    args.at_num = 1000
    args.test_num = 1000
    at_ratio = 0.5
    args.batch_size = 1000
    args.random_seed = 0

    if args.dataset == "cifar":
        args.lr = 0.07
        eps_step = 0.008
        iter_round = 20
        eps = 0.025
    elif args.dataset == "mnist":
        args.lr = 0.07
        eps_step = 0.01
        iter_round = 40
        eps = 0.2

    fix_random(args.random_seed)
    dataset_train, dataset_test = load_data(args)
    at_idxs, test_idxs = at_sample(dataset_test, at_ratio)
    print("##############################################################################")
    retrain(dataset_test, at_idxs, test_idxs)
