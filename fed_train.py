import copy
import os
import torch
import wandb
from torch import device
from armor_py.utils import fix_random, aggregate, wandb_init, local_test_on_all_clients
from armor_py.options import args_parser
from armor_py.sampling import load_data, sample_user, create_model
from armor_py.update import LocalUpdate


def train(dataset_train, dataset_test, dict_users, dict_server):
    print("dataset =", args.dataset, ", num_users =", args.client_num_in_total, ", comm_round =", args.comm_round)
    fix_random(args.random_seed)
    # start wandb
    run = wandb_init(args)
    # make directory
    model_path = "fl_model/{}/client_num_{}/".format(args.dataset, args.client_num_in_total)
    final_path = model_path + "Client_last/"
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    loss_test, loss_train, acc_test, acc_train = [], [], [], []
    Client = {}
    # create model
    net_glob = create_model(args, device)
    net_glob.train()
    w_glob = net_glob.state_dict()
    for idx in range(args.client_num_in_total):
        Client[idx] = w_glob
    # training local models
    for iter_round in range(args.comm_round):
        print('\n', '*' * 20, 'Communication Round: {}'.format(iter_round), '*' * 20)
        w_locals, loss_locals, acc_locals = [], [], []
        for idx in range(args.client_num_in_total):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], device=device)
            w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob), device=device)
            w_locals.append(w)
            loss_locals.append(loss)
            acc_locals.append(acc)
            if iter_round == args.comm_round - 1:
                Client[idx] = w_locals[idx]
        # update global weights
        w_glob = aggregate(w_locals)
        # global test
        net_glob.load_state_dict(w_glob)
        acc_global_test, loss_global_test = local_test_on_all_clients(args, net_glob, dataset_test, dict_server, device)
        # print loss
        loss_avg_locals = sum(loss_locals) / len(loss_locals)
        acc_avg_locals = sum(acc_locals) / len(acc_locals)
        loss_avg_global_test = sum(loss_global_test) / len(loss_global_test)
        acc_avg_global_test = sum(acc_global_test) / len(acc_global_test)
        # record results
        print("Global_Test/Acc =", acc_avg_global_test,
              ", Global_Test/Loss =", loss_avg_global_test,
              ", Train/Acc =", acc_avg_locals,
              ", Train/Loss =", loss_avg_locals)
        wandb.log({"Global_Test/Acc": acc_avg_global_test,
                   "Global_Test/Loss": loss_avg_global_test,
                   "Train/Acc": acc_avg_locals,
                   "Train/Loss": loss_avg_locals,
                   })
        loss_train.append(loss_avg_locals)
        acc_train.append(acc_avg_locals)
        loss_test.append(loss_avg_global_test)
        acc_test.append(acc_avg_global_test)

    # save local models of the last communication round
    for idx in range(args.client_num_in_total):
        torch.save(Client[idx], final_path + "Client_last_{}.pth".format(idx))
    # save global model
    torch.save(w_glob, model_path + "Global.pth")
    # print results
    final_train_loss = sum(loss_train) / len(loss_train)
    final_train_accuracy = sum(acc_train) / len(acc_train)
    final_test_loss = sum(loss_test) / len(loss_test)
    final_test_accuracy = sum(acc_test) / len(acc_test)
    print('\nFinal train loss:', final_train_loss)
    print('\nFinal train acc:', final_train_accuracy)
    print('\nFinal test loss:', final_test_loss)
    print('\nFinal test acc:', final_test_accuracy)
    # finish wandb
    run.finish()
    print("dataset = " + args.dataset + ", num of client = {} completed!".format(args.client_num_in_total))


if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda)
    device = torch.device("cuda:0")
    pid = os.getpid()
    print('PID No.', pid)
    args.epoch = 1
    args.batch_size = 10
    args.lr = 0.07

    if args.dataset == "cifar":
        args.comm_round = 100
    elif args.dataset == "mnist":
        args.comm_round = 50

    fix_random(args.random_seed)
    dataset_train, dataset_test = load_data(args)
    dict_train, dict_test = sample_user(args, dataset_train, dataset_test)
    print("##############################################################################")
    train(dataset_train, dataset_test, dict_train, dict_test)
