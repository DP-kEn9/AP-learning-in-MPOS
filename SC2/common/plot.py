import torch
import collections
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_data(path, p):
    with open(path + "\\" + p) as f1:
        data = json.load(f1)
    data = data["0"]
    return data


def get_itr(path):
    with open(path + "\\evaluate_itr.json") as f:
        data = json.load(f)
    data = data["0"]
    return data


def plot(mapname, different, reward):
    path = ""
    if different == "reward":
        path_0 = ""
        path_1 = ""
        path_2 = ""
        labels = ["rule_based reward", "model_based reward", "baseline reward"]
    else:
        mapname0 = mapname + "_1adv"
        mapname1 = mapname + "_2adv"
        mapname2 = mapname + "_3adv"
        path_0 = ""
        path_1 = ""
        path_2 = ""
        if "8m" not in mapname:
            labels = ["1 adv agent", "2 adv agents", "3 adv agents"]
        else:
            labels = ["1 adv agent", "2 adv agents"]
    itr_0 = get_itr(path_0)
    itr_1 = get_itr(path_1)
    min_len = min(len(itr_0), len(itr_1))
    win_rates_0 = get_data(path_0, "win_rates.json")
    win_rates_1 = get_data(path_1, "win_rates.json")
    ep_reward_0 = get_data(path_0, "episodes_rewards.json")
    ep_reward_1 = get_data(path_1, "episodes_rewards.json")
    wr_0, wr_1, writr_0, writr_1 = [], [], [], []
    for i in range(0, len(itr_0)-5):










        wr_0.append(win_rates_0[i:i+5])
        wr_1.append(win_rates_1[i:i+5])
        writr_0.append(sum(itr_0[i:i+5]) / 5)
        writr_1.append(sum(itr_1[i:i+5]) / 5)
    wr_0, wr_1, writr_0, writr_1, ep_reward_0, ep_reward_1, itr_0, itr_1 =\
        1-np.array(wr_0), 1-np.array(wr_1), np.array(writr_0), np.array(writr_1),\
            np.array(ep_reward_0), np.array(ep_reward_1), np.array(itr_0), np.array(itr_1)
    print(wr_1)
    writr_0, writr_1 = np.repeat(writr_0, 5)[:, None] / 1e6, np.repeat(writr_1, 5)[:, None] / 1e6
    wr_0, wr_1 = wr_0.flatten()[:, None], wr_1.flatten()[:, None]
    itr_0, itr_1, ep_reward_0, ep_reward_1 = itr_0[:min_len], itr_1[:min_len],\
        ep_reward_0[:min_len, :], ep_reward_1[:min_len, :]
    wr_0[wr_0 > 1] = 1
    wr_1[wr_1 > 1] = 1
    itr_0, itr_1 = np.repeat(itr_0, 50)[:, None] / 1e6, np.repeat(itr_1, 50)[:, None] / 1e6
    ep_reward_0, ep_reward_1 = ep_reward_0.flatten()[:, None], ep_reward_1.flatten()[:, None]
    fig = plt.figure()
    if different != "reward":
        ax1 = fig.add_subplot(211)
    else:
        ax1 = fig.add_subplot(111)
    plot_wr0 = pd.DataFrame(np.concatenate((writr_0, wr_0), axis=1), columns=['T (mil)', 'win_rates'])
    plot_wr1 = pd.DataFrame(np.concatenate((writr_1, wr_1), axis=1), columns=['T (mil)', 'win_rates'])
    plot_reward0 = pd.DataFrame(np.concatenate((itr_0, ep_reward_0), axis=1), columns=['T (mil)', 'Test Returns'])
    plot_reward1 = pd.DataFrame(np.concatenate((itr_1, ep_reward_1), axis=1), columns=['T (mil)', 'Test Returns'])
    sns.lineplot(x="T (mil)", y="win_rates", data=plot_wr0, ax=ax1, ci='sd', estimator=np.average, color="red")
    sns.lineplot(x="T (mil)", y="win_rates", data=plot_wr1, ax=ax1, ci='sd', estimator=np.average, color="blue")
    if different != "reward":
        ax2 = fig.add_subplot(212)
        sns.lineplot(x="T (mil)", y="Test Returns", data=plot_reward0, ax=ax2, ci='sd', estimator=np.median,
                     color="red")
        sns.lineplot(x="T (mil)", y="Test Returns", data=plot_reward1, ax=ax2, ci='sd', estimator=np.median,
                     color="blue")
    if "8m" not in mapname or different == "reward":
        itr_2 = get_itr(path_2)
        min_len = min(min_len, len(itr_2))
        win_rates_2 = get_data(path_2, "win_rates.json")
        ep_reward_2 = get_data(path_2, "episodes_rewards.json")
        wr_2, writr_2 = [], []
        for i in range(1, len(itr_2)):
            if 5 * i >= min_len:


                break
            wr_2.append(win_rates_2[5 * (i - 1):5 * i])
            writr_2.append(sum(itr_2[5 * (i - 1):5 * i]) / 5)
        wr_2, writr_2, ep_reward_2, itr_2 = np.array(wr_2), np.array(writr_2), np.array(ep_reward_2), np.array(itr_2)
        itr_2, ep_reward_2 = itr_2[:min_len], ep_reward_2[:min_len, :]
        wr_2, ep_reward_2 = wr_2.flatten()[:, None], ep_reward_2.flatten()[:, None]
        wr_2[wr_2 > 1] = 1
        writr_2, itr_2 = np.repeat(writr_2, 5)[:, None] / 1e6, np.repeat(itr_2, 50)[:, None] / 1e6
        plot_wr2 = pd.DataFrame(np.concatenate((writr_2, wr_2), axis=1), columns=['T (mil)', 'win_rates'])

        if different != "reward":
            plot_reward2 = pd.DataFrame(np.concatenate((itr_2, ep_reward_2), axis=1),
                                        columns=['T (mil)', 'Test Returns'])
            sns.lineplot(x="T (mil)", y="Test Returns", data=plot_reward2, ax=ax2, ci='sd', estimator=np.average,
                         color="green")





    plt.tight_layout()


    fig.savefig(path + "\\" + mapname + different + reward + ".png")
    return fig


def plot_8m_reward(j):
    path_0 = ""
    path_1 = ""
    path_2 = ""
    itr_0 = get_itr(path_0)
    itr_1 = get_itr(path_1)
    min_len = min(len(itr_0), len(itr_1))
    win_rates_0 = get_data(path_0, "win_rates.json")
    win_rates_1 = get_data(path_1, "win_rates.json")
    ep_reward_0 = get_data(path_0, "episodes_rewards.json")
    ep_reward_1 = get_data(path_1, "episodes_rewards.json")
    wr_0, wr_1, writr_0, writr_1 = [], [], [], []
    for i in range(1, len(itr_0)):
        if j * i >= min_len:




            break
        wr_0.append(win_rates_0[j * (i - 1):j * i])
        wr_1.append(win_rates_1[j * (i - 1):j * i])
        writr_0.append(sum(itr_0[j * (i - 1):j * i]) / j)
        writr_1.append(sum(itr_1[j * (i - 1):j * i]) / j)
    wr_0, wr_1, writr_0, writr_1, ep_reward_0, ep_reward_1, itr_0, itr_1 =\
        np.array(wr_0), np.array(wr_1), np.array(writr_0), np.array(writr_1),\
            np.array(ep_reward_0), np.array(ep_reward_1), np.array(itr_0), np.array(itr_1)
    writr_0, writr_1 = np.repeat(writr_0, j)[:, None] / 1e6, np.repeat(writr_1, j)[:, None] / 1e6
    wr_0, wr_1 = wr_0.flatten()[:, None], wr_1.flatten()[:, None]
    itr_0, itr_1, ep_reward_0, ep_reward_1 = itr_0[:min_len], itr_1[:min_len],\
        ep_reward_0[:min_len, :], ep_reward_1[:min_len, :]
    wr_0[wr_0 > 1] = 1
    wr_1[wr_1 > 1] = 1
    itr_0, itr_1 = np.repeat(itr_0, 50)[:, None] / 1e6, np.repeat(itr_1, 50)[:, None] / 1e6
    ep_reward_0, ep_reward_1 = ep_reward_0.flatten()[:, None], ep_reward_1.flatten()[:, None]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plot_wr0 = pd.DataFrame(np.concatenate((writr_0, wr_0), axis=1), columns=['T (mil)', 'win_rates'])
    plot_wr1 = pd.DataFrame(np.concatenate((writr_1, wr_1), axis=1), columns=['T (mil)', 'win_rates'])


    sns.lineplot(x="T (mil)", y="win_rates", data=plot_wr0, ax=ax1, ci='sd', estimator=np.average, color="red").set_title("8m")
    sns.lineplot(x="T (mil)", y="win_rates", data=plot_wr1, ax=ax1, ci='sd', estimator=np.average, color="green")


    fig = plot_one(path_2, j=j, fig=fig, color="blue")
    fig.savefig("")
    return fig


def plot_one(path, j=5, fig=None, color="green", map=""):
    win_rates = get_data(path, "win_rates.json")
    itr = get_itr(path)
    wr, writr = [], []
    for i in range(0, len(itr)-5):
        wr.append(win_rates[i:i+j])
        writr.append(sum(itr[i:i+j]) / j)

    wr, writr = 1-np.array(wr), np.array(writr)
    wr = wr.flatten()[:, None]
    writr = np.repeat(writr, j)[:, None] / 1e6
    plot_wr = pd.DataFrame(np.concatenate((writr, wr), axis=1), columns=['T (mil)', 'win_rates'])
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(212)
    sns.lineplot(x="T (mil)", y="win_rates", data=plot_wr, ax=ax, ci='sd', estimator=np.average, color=color).set_title(map)
    return fig


if __name__ == '__main__':
    path = ""

    fig = plot_one(path, color="red", map="intersection_3adv")
    fig.savefig("")
