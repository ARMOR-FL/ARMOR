import copy
import os
import numpy as np
import torch
from easydict import EasyDict
from attack.projected_gradient_descent import projected_gradient_descent
from armor_py.utils import del_tensor_element, fix_random
from armor_py.models import CNN_CIFAR, CNN_MNIST
from armor_py.options import args_parser
from armor_py.sampling import ld_mnist, ld_cifar10
np.set_printoptions(threshold=np.inf)


def per_pgd_attack():
    print("dataset = " + args.dataset + subfix + ", num of client = {}".format(args.client_num_in_total))
    client_num_test = args.client_num_in_total
    model_acc = {}
    prefix = "Client_at"
    pgd_data = ""
    list_data = ""
    pgd_file = path + "pgd" + subfix + ".out"
    attack_list = path + "attack_list" + subfix + ".out"

    pgd_data += "eps={:.3f}, eps_step={:.3f}, iter_round={}\n".format(eps, eps_step, iter_round)
    list_data += "eps={:.3f}, eps_step={:.3f}, iter_round={}\n".format(eps, eps_step, iter_round)

    # load clients
    net = {}
    for corrupted_idx in range(args.client_num_in_total):
        net[corrupted_idx] = copy.deepcopy(net_glob)
        file_path = path + "Client" + subfix + "/" + prefix + "_{}.pth".format(corrupted_idx)
        net[corrupted_idx].load_state_dict(torch.load(file_path))
        net[corrupted_idx].to(device)
        net[corrupted_idx].eval()

    pgd_data += "################################ Attack begin ################################\n"
    list_data += "################################ Attack begin ################################\n"
    for corrupted_idx in range(client_num_test):
        fix_random(corrupted_idx + args.random_seed)
        pgd_data += "##############################################################################\n"
        pgd_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        list_data += "##############################################################################\n"
        list_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        test_round = 0
        for images, labels in data.client:
            if test_round >= 1:
                break
            images = images.to(device)
            labels = labels.to(device)
            images_pgd = projected_gradient_descent(net[corrupted_idx], images, eps, eps_step, iter_round, np.inf)
            _, corrupt_y_pred = net[corrupted_idx](images).max(1)
            _, corrupt_y_pred_pgd = net[corrupted_idx](images_pgd).max(1)
            model_acc[corrupted_idx] = corrupt_y_pred.eq(labels).sum().item() / labels.size(0)
            pgd_data += "Model Acc of Client {}: {:.2f}(%) ************* Generated\n".format(corrupted_idx, (
                    model_acc[corrupted_idx] * 100.0))
            tensor_size = images_pgd.shape[0]
            for i in range(tensor_size):
                if corrupt_y_pred[tensor_size - 1 - i] != labels[tensor_size - 1 - i]:
                    images = del_tensor_element(images, tensor_size - 1 - i)
                    labels = del_tensor_element(labels, tensor_size - 1 - i)
                    images_pgd = del_tensor_element(images_pgd, tensor_size - 1 - i)

            _, corrupt_y_pred = net[corrupted_idx](images).max(1)
            _, corrupt_y_pred_pgd = net[corrupted_idx](images_pgd).max(1)
            print("Client", corrupted_idx, "size of clean samples predicted correctly:", images_pgd.shape[0],
                  images.shape[0], labels.shape[0])
            tensor_size = images_pgd.shape[0]
            for i in range(tensor_size):
                if corrupt_y_pred_pgd[tensor_size - 1 - i] == corrupt_y_pred[tensor_size - 1 - i]:
                    images = del_tensor_element(images, tensor_size - 1 - i)
                    labels = del_tensor_element(labels, tensor_size - 1 - i)
                    images_pgd = del_tensor_element(images_pgd, tensor_size - 1 - i)
            print("Client", corrupted_idx, "size of succeed samples:", images_pgd.shape[0], images.shape[0],
                  labels.shape[0])
            print("------------------------------------------------------------------------------")

            for idx in range(args.client_num_in_total):
                report = EasyDict(nb_test=0, nb_correct=0, correct_pgd_predict=0, correct_pgd_in_corrected=0)
                _, y_pred = net[idx](images).max(1)
                _, y_pred_pgd = net[idx](images_pgd).max(1)
                report.nb_test += labels.size(0)
                report.nb_correct += y_pred.eq(labels).sum().item()
                # 0 predict incorrectly
                # 1 predict correctly & attack failed
                # 2 predict correctly & attack succeed
                if report.nb_test != 0:
                    list_mask = y_pred.eq(labels)  # predict correctly = True
                    list_value = ~y_pred_pgd.eq(y_pred)  # attack successfully = True
                    list_result = list_mask & list_value  # predict correctly & attack succeed = True
                    list_result = (list_result).long().cpu().numpy()  # predict correctly & attack succeed
                    list_mask = list_mask.long().cpu().numpy()  # predict correctly & attack failed= 1
                    list_result = str(list_result + list_mask).replace("\n",
                                                                       "")  # predict correctly & attack succeed = 2
                    list_result = list_result.replace("[", "")
                    list_result = list_result.replace("]", "")
                    list_data += list_result + "\n"

                    y_pred_correct = y_pred
                    y_pred_correct_pgd = y_pred_pgd
                    # only consider condition where predict succeed
                    for i in range(images.shape[0]):
                        if y_pred[images.shape[0] - 1 - i] != labels[images.shape[0] - 1 - i]:
                            y_pred_correct = del_tensor_element(y_pred_correct, images.shape[0] - 1 - i)
                            y_pred_correct_pgd = del_tensor_element(y_pred_correct_pgd, images.shape[0] - 1 - i)
                    report.correct_pgd_in_corrected += y_pred_correct_pgd.eq(y_pred_correct).sum().item()

                    if idx == corrupted_idx:
                        pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%) " \
                                    "************* Generated\n".format(idx,
                            (report.nb_correct / report.nb_test * 100.0),
                            ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))
                    else:
                        pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%)\n".format(
                            idx,
                            (report.nb_correct / report.nb_test * 100.0),
                            ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))
                else:
                    if idx == corrupted_idx:
                        pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%) " \
                                    "************* Generated\n".format(idx, 0, 0)
                    else:
                        pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%)\n".format(
                            idx, 0, 0)

            pgd_data += "##############################################################################\n"
            list_data += "##############################################################################\n"
            test_round += 1

    with open(attack_list, "w", encoding="utf-8") as f:
        f.write(list_data)
    with open(pgd_file, "w", encoding="utf-8") as f:
        f.write(pgd_data)
    print("dataset = " + args.dataset + subfix + ", num of client = {} complete!".format(
            args.client_num_in_total))


if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.cuda)
    device = torch.device("cuda:0")
    pid = os.getpid()
    print('PID No.', pid)
    args.random_seed = 0
    fix_random(args.random_seed)

    if args.dataset == "cifar":
        eps_step = 0.008
        iter_round = 20
        eps = 0.025
        data = ld_cifar10(batch_size=1000)
        net_glob = CNN_CIFAR()
        net_glob.to(device)
        net_glob.eval()
    elif args.dataset == "mnist":
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        data = ld_mnist(batch_size=1000)
        net_glob = CNN_MNIST()
        net_glob.to(device)
        net_glob.eval()

    print("eps=",eps,"eps_step=",eps_step,"iter_round=",iter_round)

    subfix = "_sub_{:.2f}_eta_{}_epoch_1000_p_{:.2f}".format(args.percent_sub, args.eta, args.p)
    path = "at_model/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    per_pgd_attack()
