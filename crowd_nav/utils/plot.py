import re
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import numpy as np


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def get_data_from_file(log_file):
    with open(log_file, 'r') as file:
        log = file.read()

    val_pattern = r"VAL   in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                  r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                  r"total reward: (?P<reward>[-+]?\d+.\d+)"
    val_episode = []
    val_sr = []
    val_cr = []
    val_time = []
    val_reward = []
    for r in re.findall(val_pattern, log):
        val_episode.append(int(r[0]))
        val_sr.append(float(r[1]))
        val_cr.append(float(r[2]))
        val_time.append(float(r[3]))
        val_reward.append(float(r[4]))

    train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                    r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                    r"total reward: (?P<reward>[-+]?\d+.\d+), average return: (?P<return>[-+]?\d+.\d+)"
    train_episode = []
    train_sr = []
    train_cr = []
    train_time = []
    train_reward = []
    train_return = []
    for r in re.findall(train_pattern, log):
        train_episode.append(int(r[0]))
        train_sr.append(float(r[1]))
        train_cr.append(float(r[2]))
        train_time.append(float(r[3]))
        train_reward.append(float(r[4]))
        train_return.append(float(r[5]))
        # train_return.append(float(r[5]))
    del train_episode[4000]
    del train_reward[4000]
    del train_sr[4000]
    del train_cr[4000]
    del train_time[4000]
    del train_return[4000]
    del train_episode[8000]
    del train_reward[8000]
    del train_sr[8000]
    del train_cr[8000]
    del train_time[8000]
    del train_return[8000]
    return train_sr, train_cr, train_time, train_reward, train_return

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_sr', default=False, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_time', default=False, action='store_true')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_val', default=False, action='store_true')
    parser.add_argument('--plot_all', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=50)
    args = parser.parse_args()

    models = []
    max_episodes = None

    ax1 = ax2 = ax3 = ax4 = None
    ax1_legends = []
    ax2_legends = []
    ax3_legends = []
    ax4_legends = []

    # args.log_files = sorted(args.log_files)
    # if not models:
    #     models = [os.path.basename(log_file[:-11]) for log_file in args.log_files]

    sr_data = []
    cr_data = []
    nav_data = []
    reward_data = []
    return_data = []
    # for i, log_file in enumerate(args.log_files):
    for i in range(5):
        k = i+5
        log_file = 'data/two_layer/logs/'+str(k)+'.log'
        sr, cr, nav_time, reward, return_ = get_data_from_file(log_file)
        sr_data.append(sr)
        cr_data.append(cr)
        nav_data.append(nav_time)
        reward_data.append(reward)
        return_data.append(return_)
    sr_data = np.array(sr_data)
    cr_data = np.array(cr_data)
    nav_data = np.array(nav_data)
    reward_data = np.array(reward_data)
    reward_data = np.array(return_data)

    dim_pos=0
    reward_mean = np.mean(reward_data,axis=dim_pos)
    sr_mean = np.mean(sr_data, axis=dim_pos)
    cr_mean = np.mean(cr_data, axis=dim_pos)
    nav_mean = np.mean(nav_data, axis=dim_pos)
    reward_max = np.max(reward_data,axis=dim_pos)
    sr_max = np.max(sr_data, axis=dim_pos)
    cr_max = np.max(cr_data, axis=dim_pos)
    nav_max = np.max(nav_data, axis=dim_pos)
    reward_min = np.min(reward_data,axis=dim_pos)
    sr_min = np.min(sr_data, axis=dim_pos)
    cr_min = np.min(cr_data, axis=dim_pos)
    nav_min = np.min(nav_data, axis=dim_pos)

    # smooth training plot
    reward_mean = running_mean(reward_mean, args.window_size)
    reward_min = running_mean(reward_min, args.window_size)
    reward_max = running_mean(reward_max, args.window_size)
    episode = np.arange(1, len(reward_mean)+1)
    print("over")

    plt.fill_between(episode, reward_min, reward_max, facecolor='brown', alpha=0.3)
    plt.axvline(x=4000, color='blue', linewidth='2', linestyle='dashed')
    plt.axvline(x=8000, color='blue', linewidth='2', linestyle='dashed')
    plt.plot(episode, reward_mean, color='r', marker='.', markersize=1.5, linewidth='1', linestyle='solid')
    plt.axis([0, 15000, -1, 1.5])
    plt.xlabel('Episode')
    plt.ylabel('Averaged Return')
    savefig("reward_record.eps")
    plt.show()

    # if max_episodes is not None:
        #     train_episode = train_episode[:max_episodes]
        #     train_sr = train_sr[:max_episodes]
        #     train_cr = train_cr[:max_episodes]
        #     train_time = train_time[:max_episodes]
        #     train_reward = train_reward[:max_episodes]

        # # smooth training plot
        # train_sr_smooth = running_mean(train_sr, args.window_size)
        # train_cr_smooth = running_mean(train_cr, args.window_size)
        # train_time_smooth = running_mean(train_time, args.window_size)
        # train_reward_smooth = running_mean(train_reward, args.window_size)

        # # plot sr
        # if args.plot_sr:
        #     if ax1 is None:
        #         _, ax1 = plt.subplots()
        #     if args.plot_train:
        #         ax1.plot(range(len(train_sr_smooth)), train_sr_smooth)
        #         ax1_legends.append(models[i])
        #     if args.plot_val:
        #         ax1.plot(val_episode, val_sr)
        #         ax1_legends.append(models[i])
        #
        #     ax1.legend(ax1_legends)
        #     ax1.set_xlabel('Episodes')
        #     ax1.set_ylabel('Success Rate')
        #     ax1.set_title('Success rate')

        # # plot time
        # if args.plot_time:
        #     if ax2 is None:
        #         _, ax2 = plt.subplots()
        #     if args.plot_train:
        #         ax2.plot(range(len(train_time_smooth)), train_time_smooth)
        #         ax2_legends.append(models[i])
        #     if args.plot_val:
        #         ax2.plot(val_episode, val_time)
        #         ax2_legends.append(models[i])
        #
        #     ax2.legend(ax2_legends)
        #     ax2.set_xlabel('Episodes')
        #     ax2.set_ylabel('Time(s)')
        #     ax2.set_title("Robot's Time to Reach Goal")

        # # plot cr
        # if args.plot_cr:
        #     if ax3 is None:
        #         _, ax3 = plt.subplots()
        #     if args.plot_train:
        #         ax3.plot(range(len(train_cr_smooth)), train_cr_smooth)
        #         ax3_legends.append(models[i])
        #     if args.plot_val:
        #         ax3.plot(val_episode, val_cr)
        #         ax3_legends.append(models[i])
        #
        #     ax3.legend(ax3_legends)
        #     ax3.set_xlabel('Episodes')
        #     ax3.set_ylabel('Collision Rate')
        #     ax3.set_title('Collision Rate')

        # plot reward
        # if args.plot_reward:
        #     if ax4 is None:
        #         _, ax4 = plt.subplots()
        #     if args.plot_train:
        #         ax4.plot(range(len(train_reward_smooth)), train_reward_smooth)
        #         ax4_legends.append(models[i])
        #     if args.plot_val:
        #         ax4.plot(val_episode, val_reward)
        #         ax4_legends.append(models[i])

        # if args.plot_sr:
        #     ax1.legend(ax1_legends)
        #     ax1.set_title('Success rate')
        #
        # if args.plot_time:
        #     ax2.legend(ax2_legends)
        #     ax2.set_title("robot's Time to Reach Goal")
        #
        # if args.plot_cr:
        #     ax3.legend(ax3_legends)
        #     ax3.set_title('Collision Rate')

        # if args.plot_reward:
        #     # ax4.legend(ax4_legends, loc='center left', bbox_to_anchor=(1, 0.5))
        #     ax4.legend(ax4_legends)
        #     ax4.set_xlabel('Episodes')
        #     ax4.set_ylabel('Reward')
        #     plt.tick_params(axis='both', which='major')
        #     plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.125)
        #     # ax4.set_xlabel('xlabel', fontsize=18)
        #     # ax4.set_ylabel('ylabel', fontsize=16)
        #     # ax4.set_title('Cumulative Discounted Reward')


if __name__ == '__main__':
    main()
