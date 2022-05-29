import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from armor_py.options import args_parser
from armor_py.utils import alter_re, alter, del_blank_line, dict_avg, test_remove

def asr_per_process():
    client_num_in_total = args.client_num_in_total
    path = dataset_path + "client_num_{}/".format(client_num_in_total)
    file_path = path + prefix_pgd + model_name + ".out"
    pure_acc_file_path = path + prefix_pgd + model_name + "_pure_acc.out"
    shutil.copyfile(file_path, pure_acc_file_path)
    acc_file_path = path + prefix_pgd + model_name + "_acc.out"
    shutil.copyfile(file_path, acc_file_path)
    asr_file_path = path + prefix_pgd + model_name + "_asr.out"
    shutil.copyfile(file_path, asr_file_path)

    ### Pure Acc ###
    alter_re(pure_acc_file_path, "eps=.*", "")
    alter(pure_acc_file_path, "################################ Attack begin ################################", "")
    alter(pure_acc_file_path, "##############################################################################", "")
    alter_re(pure_acc_file_path, "Adversary Examples Generated on Client .*", "")
    alter_re(pure_acc_file_path, "Model Acc of Client .*: ", "")
    alter_re(pure_acc_file_path, "Test on Client .*", "")
    alter_re(pure_acc_file_path, "\(%\).*", "")
    del_blank_line(pure_acc_file_path)

    ### Acc ###
    alter_re(acc_file_path, "eps=.*", "")
    alter(acc_file_path, "################################ Attack begin ################################", "")
    alter(acc_file_path, "##############################################################################", "")
    alter(acc_file_path, "Adversary Examples Generated on Client ", "")
    alter_re(acc_file_path, "Test on Client .* Generated", "")
    alter_re(acc_file_path, "Test on Client .* Acc: ", "")
    alter_re(acc_file_path, "Model Acc of Client .*", "")
    alter_re(acc_file_path, "\(%\).*", "")
    del_blank_line(acc_file_path)

    ### ASR ###
    alter_re(asr_file_path, "eps=.*", "")
    alter(asr_file_path, "################################ Attack begin ################################", "")
    alter(asr_file_path, "##############################################################################", "")
    alter(asr_file_path, "Adversary Examples Generated on Client ", "")
    alter_re(asr_file_path, "Test on Client .* Generated", "")
    alter_re(asr_file_path, "Test on Client .* ASR: ", "")
    alter_re(asr_file_path, "Model Acc of Client .*", "")
    alter_re(asr_file_path, "\(%\).*", "")
    del_blank_line(asr_file_path)

    result_path = dataset_path + "ASR/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = result_path + args.dataset + "_client_num_{}".format(client_num_in_total) + model_name + ".out"

    file_data = "Client\tPure Acc\n"
    pure_acc_file = open(pure_acc_file_path)
    pure_acc = []
    for i in pure_acc_file:
        pure_acc.append(float(i))
    for corrupted_idx in range(client_num_in_total):
        file_data += "{}\t{:.2f}%\n".format(corrupted_idx, pure_acc[corrupted_idx])
    pure_acc_avg = np.average(pure_acc)
    file_data += "Average Pure Acc {:.2f}%\n".format(pure_acc_avg)

    file_data += "\n\nClient\tAcc of AE\n"
    acc_file = open(acc_file_path)
    acc, acc_avg = {}, {}
    i_idx = 0
    for i in acc_file:
        if i_idx % (client_num_in_total) == 0:
            corrupted_idx = int(i)
            acc[corrupted_idx] = []
        else:
            acc[corrupted_idx].append(float(i))
        i_idx = i_idx + 1
    for corrupted_idx in range(client_num_in_total):
        acc_avg[corrupted_idx] = np.average(acc[corrupted_idx])
        file_data += "{}\t{:.2f}%\n".format(corrupted_idx, acc_avg[corrupted_idx])
    acc_avg_avg = dict_avg(acc_avg)
    file_data += "Average Acc of AE {:.2f}%\n".format(acc_avg_avg)

    file_data += "\n\nClient\tASR of AE\n"
    asr_file = open(asr_file_path)
    asr, asr_avg = {}, {}
    i_idx = 0
    for i in asr_file:
        if i_idx % (client_num_in_total) == 0:
            corrupted_idx = int(i)
            asr[corrupted_idx] = []
        else:
            asr[corrupted_idx].append(float(i))
        i_idx = i_idx + 1
    for corrupted_idx in range(client_num_in_total):
        asr_avg[corrupted_idx] = np.average(asr[corrupted_idx])
        file_data += "{}\t{:.2f}%\n".format(corrupted_idx, asr_avg[corrupted_idx])
    asr_avg_avg = dict_avg(asr_avg)
    file_data += "Average ASR of AE {:.2f}%\n".format(asr_avg_avg)

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(file_data)

    test_remove(pure_acc_file_path)
    test_remove(acc_file_path)
    test_remove(asr_file_path)

    return pure_acc_avg, acc_avg_avg, asr_avg_avg


def atr_per_process():
    client_num_in_total = args.client_num_in_total
    path = dataset_path + "client_num_{}/".format(client_num_in_total)
    file_path = path + prefix_attack_list + model_name + ".out"
    processed_file_path = path + prefix_attack_list + model_name + "_processed.out"
    shutil.copyfile(file_path, processed_file_path)

    alter_re(processed_file_path, "eps=.*", "")
    alter(processed_file_path, "################################ Attack begin ################################", "")
    alter(processed_file_path, "##############################################################################", "")
    alter(processed_file_path, "Adversary Examples Generated on Client ", "")
    del_blank_line(processed_file_path)

    result_path = dataset_path + "ATR/out/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = result_path + args.dataset + "_client_num_{}".format(client_num_in_total) + model_name + ".out"
    file_data = "Client\tATR\n"
    file_attack_list = open(processed_file_path)

    # 0 predict incorrectly
    # 1 predict correctly & attack failed
    # 2 predict correctly & attack succeed
    num_items = {}
    raw_arr = {}
    i_idx = 0
    for i in file_attack_list:
        if i_idx % (client_num_in_total + 1) == 0:
            corrupted_idx = int(i)
        elif i_idx % (client_num_in_total + 1) == 1:
            num_items[corrupted_idx] = len(i.split())
            raw_arr[corrupted_idx] = np.zeros((client_num_in_total, num_items[corrupted_idx]))
            raw_arr[corrupted_idx][i_idx % (client_num_in_total + 1) - 1] = i.split()
        else:
            raw_arr[corrupted_idx][i_idx % (client_num_in_total + 1) - 1] = i.split()
        i_idx = i_idx + 1

    num_incorrect, num_attack_fail, num_attack_succeed, num_predict_correct = {}, {}, {}, {}
    TR, ATR, AATR = {}, {}, {}
    for corrupted_idx in range(client_num_in_total):
        num_image_used = num_items[corrupted_idx]
        num_incorrect[corrupted_idx], num_attack_fail[corrupted_idx], num_attack_succeed[corrupted_idx], \
        num_predict_correct[corrupted_idx], TR[corrupted_idx] = [], [], [], [], []
        for image_idx_used in range(num_image_used):
            num_incorrect[corrupted_idx].append(
                np.equal(raw_arr[corrupted_idx][:, image_idx_used], np.zeros(client_num_in_total)).sum())
            num_attack_fail[corrupted_idx].append(
                np.equal(raw_arr[corrupted_idx][:, image_idx_used], np.ones(client_num_in_total)).sum())
            num_attack_succeed[corrupted_idx].append(
                np.equal(raw_arr[corrupted_idx][:, image_idx_used], 2 * np.ones(client_num_in_total)).sum())
            num_predict_correct[corrupted_idx].append(
                num_attack_fail[corrupted_idx][image_idx_used] + num_attack_succeed[corrupted_idx][image_idx_used])
            TR[corrupted_idx].append(
                num_attack_succeed[corrupted_idx][image_idx_used] / num_predict_correct[corrupted_idx][image_idx_used])

        ATR[corrupted_idx] = np.average(TR[corrupted_idx])
        file_data += "{}\t{:.2f}%\n".format(corrupted_idx, ATR[corrupted_idx] * 100)

    AATR = dict_avg(ATR)
    file_data += "\nAATR\t{:.2f}%\n".format(AATR * 100)

    TR_array = []
    for TR_idx in range(len(TR)):
        TR_array.append(np.array(TR[TR_idx]))
    TR_flatten = np.hstack(TR_array)

    fontsize_ticks = 22
    fontsize_label = 26
    fontsize_legend = 18
    linewidth = 1.5

    plt.figure()
    bins = 10
    plt.xlabel("ATR on benign", fontsize=fontsize_label)
    plt.ylabel("Cumulative probability", fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='-.')
    plt.tight_layout()

    plt.hist(TR_flatten, bins, range=(0, 1), density=True, histtype='step', cumulative=True, linewidth=linewidth,
             label="ATR on benign")
    plt.legend(loc='lower right', fontsize=fontsize_legend)
    fig_path = dataset_path + "ATR/cdf/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + "cdf_client_num_{}".format(client_num_in_total) + model_name + ".pdf")
    plt.close()

    weights = np.zeros_like(TR_flatten) + 1. / TR_flatten.size
    plt.figure()
    bins = 10
    plt.xlabel("ATR on benign", fontsize=fontsize_label)
    plt.ylabel("Frequency of samples", fontsize=fontsize_label)
    plt.ylim(0, 0.6)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True, linestyle='-.')
    plt.tight_layout()

    plt.hist(TR_flatten, bins, range=(0, 1), density=False, weights=weights, alpha=0.6, label="ATR on benign")
    plt.legend(loc='upper right', fontsize=fontsize_legend)
    fig_path = dataset_path + "ATR/pdf/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + "pdf_client_num_{}".format(client_num_in_total) + model_name + ".pdf")
    plt.close()

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(file_data)
    test_remove(processed_file_path)
    return AATR


if __name__ == '__main__':
    args = args_parser()
    prefix_pgd = "pgd"
    prefix_attack_list = "attack_list"

    model_name = "_sub_{:.2f}_eta_{}_epoch_1000_p_{:.2f}".format(args.percent_sub, args.eta, args.p)
    path = "./at_model/"
    dataset_path = path + args.dataset + "/"
    # rcd_path = path + "rcd_" + args.dataset + model_name + "_num_{}".format(args.client_num_in_total) + ".out"
    # rcd_data = "num\tall_acc\tclean_acc\tasr\taatr\n"
    pure_acc_avg, acc_avg_avg, asr_avg_avg = asr_per_process()
    aatr = atr_per_process()
    print("num={}  all_acc={:.2f}%  clean_acc={:.2f}%  asr={:.2f}%  aatr={:.2f}%".format(args.client_num_in_total, pure_acc_avg,
                                                                  acc_avg_avg, asr_avg_avg, aatr * 100))
    # rcd_data += "{}\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\n".format(args.client_num_in_total, pure_acc_avg,
    #                                                               acc_avg_avg, asr_avg_avg, aatr * 100)
    # print("dataset = " + args.dataset + ", num of client = {}, model{} completed!".format(args.client_num_in_total,
    #                                                                                  model_name))
    # with open(rcd_path, "w", encoding="utf-8") as f:
    #     f.write(rcd_data)