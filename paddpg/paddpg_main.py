import argparse
import os
import time
import gym
from gym.spaces import Tuple, Discrete, Box
from collections import deque
import numpy as np
import pandas as pd
from paddpg import PADDPG

import warnings

warnings.filterwarnings("ignore")


def stack_frames(stacked_frames, frame, is_new_episode=False):
    if is_new_episode:
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    state = np.stack(stacked_frames, axis=0).reshape(-1)
    return state, stacked_frames


def choose_init_base_bid(config, budget_para):
    base_bid_path = os.path.join('../lin/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before train PAB')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_para'] == budget_para].iloc[0]['base_bid']
    avg_pctr = data[data['budget_para'] == budget_para].iloc[0]['average_pctr']

    return avg_pctr, base_bid


def get_budget(data):
    _ = []
    for day in data['day'].unique():
        current_day_budget = sum(data[data['day'].isin([day])]['market_price'])
        _.append(current_day_budget)

    return _


def reward_func(reward_type, lin_result, rl_result):
    clks = rl_result['win_clks']
    hb_clks = lin_result['win_clks']
    cost = rl_result['spend']
    hb_cost = lin_result['spend']
    pctrs = rl_result['win_pctr']

    # clk_result = clks / hb_clks if hb_clks else 1
    # cost_result = cost / hb_cost if hb_cost else 1
    # a = 1
    # b = 1
    # if reward_type == 'op':
    #     return a * clk_result - b * cost_result

    if clks >= hb_clks and cost < hb_cost:
        r = 5
    elif clks >= hb_clks and cost >= hb_cost:
        r = 1
    elif clks < hb_clks and cost >= hb_cost:
        r = -5
    else:
        r = -2.5

    if reward_type == 'op':
        return r / 1000
    elif reward_type == 'nop':
        return r
    elif reward_type == 'nop_2.0':
        return clks / 1000
    elif reward_type == 'pctr':
        return pctrs
    else:
        return clks


def bid(data, budget, **cfg):
    bid_imps = 0
    bid_clks = 0
    bid_pctr = 0
    win_imps = 0
    win_clks = 0
    win_pctr = 0
    spend = 0
    bid_action = []

    if len(data) == 0:
        return {
            'bid_imps': bid_imps,
            'bid_clks': bid_clks,
            'bid_pctr': bid_pctr,
            'win_imps': win_imps,
            'win_clks': win_clks,
            'win_pctr': win_pctr,
            'spend': spend,
            'bid_action': bid_action
        }

    data['bid_price'] = np.clip(np.multiply(data['pctr'], cfg['slot_bid_para']).astype(int), 0, 300)
    data['win'] = data.apply(lambda x: 1 if x['bid_price'] >= x['market_price'] else 0, axis=1)
    win_data = data[data['win'] == 1]
    bid_action.extend(data.values.tolist())

    if len(win_data) == 0:
        return {
            'bid_imps': len(data),
            'bid_clks': sum(data['clk']),
            'bid_pctr': sum(data['pctr']),
            'win_imps': win_imps,
            'win_clks': win_clks,
            'win_pctr': win_pctr,
            'spend': spend,
            'bid_action': bid_action
        }

    win_data['cumsum'] = win_data['market_price'].cumsum()

    if win_data.iloc[-1]['cumsum'] > budget:
        win_data = win_data[win_data['cumsum'] <= budget]

    bid_imps = len(data)
    bid_clks = sum(data['clk'])
    bid_pctr = sum(data['pctr'])
    win_imps = len(win_data)
    win_clks = sum(win_data['clk'])
    win_pctr = sum(win_data['pctr'])
    spend = sum(win_data['market_price'])

    return {
        'bid_imps': bid_imps,
        'bid_clks': bid_clks,
        'bid_pctr': bid_pctr,
        'win_imps': win_imps,
        'win_clks': win_clks,
        'win_pctr': win_pctr,
        'spend': spend,
        'bid_action': bid_action
    }


def rtb(data, budget_para, RL, config, train=True):
    if train:
        RL.is_test = False
    else:
        RL.is_test = True

    time_fraction = config['time_fraction']

    budget = get_budget(data)
    budget = np.divide(budget, budget_para)

    avg_pctr, base_bid = choose_init_base_bid(config, budget_para)

    episode_bid_imps = []
    episode_bid_clks = []
    episode_bid_pctr = []
    episode_win_imps = []
    episode_win_clks = []
    episode_win_pctr = []
    episode_spend = []
    episode_bid_action = []

    episode_action = []
    episode_reward = []

    for day_index, day in enumerate(data['day'].unique()):
        day_bid_imps = []
        day_bid_clks = []
        day_bid_pctr = []
        day_win_imps = []
        day_win_clks = []
        day_win_pctr = []
        day_spend = []
        day_bid_action = []

        day_action = []
        day_reward = []

        day_data = data[data['day'].isin([day])]
        day_budget = [budget[day_index]]

        stacked_frames = deque([np.zeros(6) for _ in range(4)], maxlen=4)

        for slot in range(0, time_fraction):
            if slot == 0:
                frame = [avg_pctr, 0, 0, 0, 0, 0]
                state, stacked_frames = stack_frames(stacked_frames, frame, is_new_episode=True)
            else:
                # left_slot_ratio = (time_fraction - 1 - slot) / (time_fraction - 1)
                # last_slot_data = day_data[day_data['time_fraction'] == slot - 1]
                # last_slot_avg_pctr = last_slot_data['pctr'].sum() / len(last_slot_data) if len(last_slot_data) else 0
                # frame = [
                #     last_slot_avg_pctr,
                #     (day_budget[-1] / day_budget[0]) / left_slot_ratio if left_slot_ratio else day_budget[-1] /
                #                                                                                day_budget[0],
                #     day_spend[-1] / day_budget[0],
                #     day_action[-1],
                #     day_win_clks[-1] / day_win_imps[-1] if day_win_imps[-1] else 0,
                #     day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] else 0
                # ]
                # state = stack_frames(stacked_frames, frame)
                state = next_state

            act, act_param, all_actions, all_action_parameters = RL.select_action(state)

            slot_lin_bid_para = base_bid / avg_pctr

            if act:
                day_action.append(act_param[0])
                slot_rl_bid_para = slot_lin_bid_para / (1 + act_param[0])
            else:
                day_action.append(-act_param[0])
                slot_rl_bid_para = slot_lin_bid_para / (1 - act_param[0])

            slot_data = day_data[day_data['time_fraction'] == slot]

            slot_lin_result = bid(slot_data, day_budget[-1], slot_bid_para=slot_lin_bid_para)
            slot_rl_result = bid(slot_data, day_budget[-1], slot_bid_para=slot_rl_bid_para)

            slot_reward = reward_func(config['reward_type'], slot_lin_result, slot_rl_result)
            day_reward.append(slot_reward)

            day_bid_imps.append(slot_rl_result['bid_imps'])
            day_bid_clks.append(slot_rl_result['bid_clks'])
            day_bid_pctr.append(slot_rl_result['bid_pctr'])
            day_win_imps.append(slot_rl_result['win_imps'])
            day_win_clks.append(slot_rl_result['win_clks'])
            day_win_pctr.append(slot_rl_result['win_pctr'])
            day_spend.append(slot_rl_result['spend'])
            day_budget.append(day_budget[-1] - slot_rl_result['spend'])
            day_bid_action.extend(slot_rl_result['bid_action'])

            if slot == time_fraction - 1:
                done = 1
                day_budget.pop(-1)
            else:
                done = 0

            left_slot_ratio = (time_fraction - 2 - slot) / (time_fraction - 1)
            slot_avg_pctr = slot_data['pctr'].sum() / len(slot_data) if len(slot_data) else 0
            next_frame = [
                slot_avg_pctr,
                (day_budget[-1] / day_budget[0]) / left_slot_ratio if left_slot_ratio else day_budget[-1] /
                                                                                           day_budget[0],
                day_spend[-1] / day_budget[0],
                day_action[-1],
                day_win_clks[-1] / day_win_imps[-1] if day_win_imps[-1] else 0,
                day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] else 0
            ]
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame)

            if train:
                RL.store(slot_reward, next_state, done)
                if len(RL.memory) >= RL.batch_size and RL.total_step > RL.initial_random_steps:
                    critic_loss = RL.update_model()
                    global critic_loss_cnt
                    critic_loss_cnt += 1
                    RL.writer.add_scalar('critic_loss', critic_loss, critic_loss_cnt)

            if done:
                break

        RL.update_epsilon()

        episode_bid_imps.append(sum(day_bid_imps))
        episode_bid_clks.append(sum(day_bid_clks))
        episode_bid_pctr.append(sum(day_bid_pctr))
        episode_win_imps.append(sum(day_win_imps))
        episode_win_clks.append(sum(day_win_clks))
        episode_win_pctr.append(sum(day_win_pctr))
        episode_spend.append(sum(day_spend))
        episode_bid_action.extend(day_bid_action)

        episode_action.append(day_action)
        episode_reward.append(sum(day_reward))

    if train:
        result = "训练"
    else:
        result = "测试"

    print(
        result + "：点击数 {}, 真实点击数 {}, pCTR {:.4f}, 真实pCTR {:.4f}, 赢标数 {}, 真实曝光数 {}, 花费 {}, CPM {:.4f}, CPC {:.4f}, 奖励 {:.2f}".format(
            int(sum(episode_win_clks)),
            int(sum(episode_bid_clks)),
            sum(episode_win_pctr),
            sum(episode_bid_pctr),
            sum(episode_win_imps),
            sum(episode_bid_imps),
            sum(episode_spend),
            sum(episode_spend) / sum(episode_win_imps),
            sum(episode_spend) / sum(episode_win_clks),
            sum(episode_reward)
        )
    )

    if train:
        global train_reward_cnt
        train_reward_cnt += 1
        RL.writer.add_scalar('train_reward', sum(episode_reward), train_reward_cnt)
    else:
        global test_reward_cnt
        test_reward_cnt += 1
        RL.writer.add_scalar('test_reward', sum(episode_reward), test_reward_cnt)

    episode_record = [
        int(sum(episode_win_clks)),
        int(sum(episode_bid_clks)),
        sum(episode_win_pctr),
        sum(episode_bid_pctr),
        sum(episode_win_imps),
        sum(episode_bid_imps),
        sum(episode_spend),
        sum(episode_spend) / sum(episode_win_imps),
        sum(episode_spend) / sum(episode_win_clks),
        sum(episode_reward)
    ]
    return episode_record, episode_action, episode_bid_action


def main(budget_para, RL, config):
    record_path = os.path.join(config['result_path'], config['campaign_id'])
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'train.bid.lin.csv'))
    test_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], 'test.bid.lin.csv'))

    header = ['clk', 'pctr', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    train_data = train_data[header]
    train_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']
    test_data = test_data[header]
    test_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']

    epoch_train_record = []
    epoch_train_action = []

    epoch_test_record = []
    epoch_test_action = []

    for epoch in range(config['train_epochs']):
        print('第{}轮'.format(epoch + 1))
        train_record, train_action, train_bid_action = rtb(train_data, budget_para, RL, config)
        test_record, test_action, test_bid_action = rtb(test_data, budget_para, RL, config, train=False)

        epoch_train_record.append(train_record)
        epoch_train_action.append(train_action)

        epoch_test_record.append(test_record)
        epoch_test_action.append(test_action)

        if config['save_bid_action']:
            bid_action_path = os.path.join(record_path, 'bid_action')
            if not os.path.exists(bid_action_path):
                os.makedirs(bid_action_path)

            train_bid_action_df = pd.DataFrame(data=train_bid_action,
                                               columns=['clk', 'pctr', 'market_price', 'day', 'time_fraction',
                                                        'bid_price', 'win'])
            train_bid_action_df.to_csv(bid_action_path + '/train_' + str(budget_para) + '_' + str(epoch) + '.csv',
                                       index=False)

            test_bid_action_df = pd.DataFrame(data=test_bid_action,
                                              columns=['clk', 'pctr', 'market_price', 'day', 'time_fraction',
                                                       'bid_price', 'win'])
            test_bid_action_df.to_csv(bid_action_path + '/test_' + str(budget_para) + '_' + str(epoch) + '.csv',
                                      index=False)

    columns = ['clks', 'real_clks', 'pctr', 'real_pctr', 'imps', 'real_imps', 'spend', 'CPM', 'CPC', 'reward']

    train_record_df = pd.DataFrame(data=epoch_train_record, columns=columns)
    train_record_df.to_csv(record_path + '/train_episode_results_' + str(budget_para) + '.csv')

    train_action_df = pd.DataFrame(data=epoch_train_action)
    train_action_df.to_csv(record_path + '/train_episode_actions_' + str(budget_para) + '.csv')

    test_record_df = pd.DataFrame(data=epoch_test_record, columns=columns)
    test_record_df.to_csv(record_path + '/test_episode_results_' + str(budget_para) + '.csv')

    test_action_df = pd.DataFrame(data=epoch_test_action)
    test_action_df.to_csv(record_path + '/test_episode_actions_' + str(budget_para) + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--feature_num', type=int, default=24)
    parser.add_argument('--action_num', type=int, default=2)
    parser.add_argument('--action_parameters_num', type=int, default=2)
    parser.add_argument('--budget_para', nargs='+', default=[2])
    parser.add_argument('--train_epochs', type=int, default=1500)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_bid_action', type=bool, default=False)
    parser.add_argument('--reward_type', type=str, default='op', help='op, nop_2.0, clk')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    config = vars(args)

    str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    config['result_path'] = config['result_path'] + '-' + str_time
    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])

    budget_para_list = list(map(int, config['budget_para']))

    critic_loss_cnt = 0
    train_reward_cnt = 0
    test_reward_cnt = 0

    obs_space = Box(
        low=0.,
        high=1.,
        shape=(config['feature_num'],),
        dtype=np.float32
    )

    action_space = Tuple((
        Discrete(2),
        Box(0, 1, (1,), np.float32),
        Box(0, 1, (1,), np.float32)
    ))

    for i in budget_para_list:
        RL = PADDPG(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=config['batch_size'],
            memory_size=config['memory_size'],
            seed=config['seed'],
            time=str_time
        )

        print('当前预算条件{}'.format(i))
        main(i, RL, config)
