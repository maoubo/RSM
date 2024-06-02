import gym
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import csv
import datetime

def make_env(env_id, seed):
    def thunk():
        if env_id == "MountainCar-v0":
            env = gym.make(env_id).unwrapped
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Collect data for each episode and record it into info
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def performance_normalization(performance, performance_max, performance_min):
    return np.clip(round((performance - performance_min) / (performance_max - performance_min), 4), 0, 1)

def calculate_ne(update, update_trans, trans_begin, per_thre):
    # This function is used to calculate the expected performance for NTP and ASR.
    if update_trans == 0:
        per_ne = 0
    else:
        sche_ne = min((update - trans_begin) / (update_trans - trans_begin), 1)
        per_ne = per_thre * sche_ne
    return per_ne

def save_results(args, result_normal_per, result_backdoor_asr, execution_time):
    file_path = '{}/{}.csv'.format(args.results_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d %H_%M')}")

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        header = ['normal_per']
        for i in range(len(result_backdoor_asr[0])):
            header.append(f'backdoor_asr_{i + 1}')
        header.append('execution_time')

        csv_writer.writerow(header)

        for a_i, b_i, c_i in zip(result_normal_per, result_backdoor_asr, execution_time):
            row = [a_i]
            row.extend(b_i)
            row.append(c_i)
            csv_writer.writerow(row)