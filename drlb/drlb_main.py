import argparse
import os

import numpy as np
import pandas as pd
from dqn import DQN

import warnings

warnings.filterwarnings("ignore")


def choose_init_lambda(config, budget_para):
    base_bid_path = os.path.join('../lin/result/ipinyou/{}/normal/test'.format(config['campaign_id']),
                                 'test_bid_log.csv')
    if not os.path.exists(base_bid_path):
        raise FileNotFoundError('Run LIN first before you train drlb')
    data = pd.read_csv(base_bid_path)
    base_bid = data[data['budget_para'] == budget_para].iloc[0]['base_bid']
    avg_pctr = data[data['budget_para'] == budget_para].iloc[0]['average_pctr']

    init_lambda = avg_pctr / base_bid

    return init_lambda


def get_budget(data):
    _ = []
    for day in data['day'].unique():
        current_day_budget = sum(data[data['day'].isin([day])]['market_price'])
        _.append(current_day_budget)

    return _


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

    data['bid_price'] = np.clip(np.divide(data['pctr'], cfg['slot_lambda']).astype(int), 0, 300)
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

    init_lambda = choose_init_lambda(config, budget_para)

    episode_bid_imps = []
    episode_bid_clks = []
    episode_bid_pctr = []
    episode_win_imps = []
    episode_win_clks = []
    episode_win_pctr = []
    episode_spend = []
    episode_bid_action = []

    episode_action = []
    episode_lambda = []
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

        day_action = [0]
        day_lambda = [init_lambda]
        day_reward = [0]

        day_data = data[data['day'].isin([day])]
        day_budget = [budget[day_index]]

        if train:
            day_state_action = []

        # 0 slot
        slot_data = day_data[day_data['time_fraction'] == 0]
        slot_lambda = init_lambda
        slot_result = bid(slot_data, day_budget[-1], slot_lambda=slot_lambda)

        day_bid_imps.append(slot_result['bid_imps'])
        day_bid_clks.append(slot_result['bid_clks'])
        day_bid_pctr.append(slot_result['bid_pctr'])
        day_win_imps.append(slot_result['win_imps'])
        day_win_clks.append(slot_result['win_clks'])
        day_win_pctr.append(slot_result['win_pctr'])
        day_spend.append(slot_result['spend'])
        day_budget.append(day_budget[-1] - slot_result['spend'])
        day_bid_action.extend(slot_result['bid_action'])

        for slot in range(1, time_fraction):
            if train and len(RL.reward_memory) >= RL.batch_size:
                reward_loss = RL.update_reward()
                global reward_loss_cnt
                reward_loss_cnt += 1
                RL.writer.add_scalar('reward_loss', reward_loss, reward_loss_cnt)

            state = [
                slot,
                day_budget[-1] / day_budget[0],
                time_fraction - slot,
                (day_budget[-1] - day_budget[-2]) / day_budget[-2] if day_budget[-2] > 0 else 0,
                day_spend[-1] / day_win_imps[-1] if day_win_imps[-1] > 0 else 0,
                day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] > 0 else 0,
                day_win_pctr[-1]
            ]

            action_index = RL.select_action(state)
            slot_action = RL.action_space[action_index]
            day_action.append(slot_action)

            slot_lambda = slot_lambda * (1 + slot_action)
            day_lambda.append(slot_lambda)

            slot_reward = RL.get_reward(state, slot_action)[0][0]
            # print(state, slot_action, slot_reward)
            day_reward.append(slot_reward)

            slot_data = day_data[day_data['time_fraction'] == slot]

            slot_result = bid(slot_data, day_budget[-1], slot_lambda=slot_lambda)
            day_bid_imps.append(slot_result['bid_imps'])
            day_bid_clks.append(slot_result['bid_clks'])
            day_bid_pctr.append(slot_result['bid_pctr'])
            day_win_imps.append(slot_result['win_imps'])
            day_win_clks.append(slot_result['win_clks'])
            day_win_pctr.append(slot_result['win_pctr'])
            day_spend.append(slot_result['spend'])
            day_budget.append(day_budget[-1] - slot_result['spend'])
            day_bid_action.extend(slot_result['bid_action'])

            if slot == time_fraction - 1:
                done = 1
                day_budget.pop(-1)
            else:
                done = 0

            next_state = [
                slot + 1,
                day_budget[-1] / day_budget[0],
                time_fraction - 1 - slot,
                (day_budget[-1] - day_budget[-2]) / day_budget[-2] if day_budget[-2] > 0 else 0,
                day_spend[-1] / day_win_imps[-1] if day_win_imps[-1] > 0 else 0,
                day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] > 0 else 0,
                day_win_pctr[-1]
            ]

            if train:
                day_state_action.append((state, slot_action))
                RL.store(slot_reward, next_state, done)
                if len(RL.memory) >= RL.batch_size:
                    model_loss = RL.update_model()
                    global model_loss_cnt
                    model_loss_cnt += 1
                    RL.writer.add_scalar('model_loss', model_loss, model_loss_cnt)

            if done:
                break

        if train:
            for (s, a) in day_state_action:
                state_action = tuple(np.append(s, a))
                max_reward = max(
                    RL.state_action_reward.get(state_action, 0),
                    sum(day_win_pctr)
                )
                RL.state_action_reward[state_action] = max_reward
                RL.reward_memory.store(s, a, max_reward, 0, 0)

        episode_bid_imps.append(sum(day_bid_imps))
        episode_bid_clks.append(sum(day_bid_clks))
        episode_bid_pctr.append(sum(day_bid_pctr))
        episode_win_imps.append(sum(day_win_imps))
        episode_win_clks.append(sum(day_win_clks))
        episode_win_pctr.append(sum(day_win_pctr))
        episode_spend.append(sum(day_spend))
        episode_bid_action.extend(day_bid_action)

        episode_action.append(day_action)
        episode_lambda.append(day_lambda)
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
    return episode_record, episode_action, episode_lambda, episode_bid_action


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
    epoch_train_lambda = []
    epoch_test_record = []
    epoch_test_action = []
    epoch_test_lambda = []

    for epoch in range(config['train_epochs']):
        print('第{}轮'.format(epoch + 1))
        train_record, train_action, train_lambda, train_bid_action = rtb(train_data, budget_para, RL, config)
        test_record, test_action, test_lambda, test_bid_action = rtb(test_data, budget_para, RL, config, train=False)

        epoch_train_record.append(train_record)
        epoch_train_action.append(train_action)
        epoch_train_lambda.append(train_lambda)

        epoch_test_record.append(test_record)
        epoch_test_action.append(test_action)
        epoch_test_lambda.append(test_lambda)

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

    train_lambda_df = pd.DataFrame(data=epoch_train_lambda)
    train_lambda_df.to_csv(record_path + '/train_episode_lambdas_' + str(budget_para) + '.csv')

    test_record_df = pd.DataFrame(data=epoch_test_record, columns=columns)
    test_record_df.to_csv(record_path + '/test_episode_results_' + str(budget_para) + '.csv')

    test_action_df = pd.DataFrame(data=epoch_test_action)
    test_action_df.to_csv(record_path + '/test_episode_actions_' + str(budget_para) + '.csv')

    test_lambda_df = pd.DataFrame(data=epoch_test_lambda)
    test_lambda_df.to_csv(record_path + '/test_episode_lambdas_' + str(budget_para) + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='3476')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--e_greedy', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--feature_num', type=int, default=7)
    parser.add_argument('--action_num', type=int, default=7)
    parser.add_argument('--budget_para', nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--train_epochs', type=int, default=1500)
    parser.add_argument('--replace_target_iter', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_bid_action', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    config = vars(args)

    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])

    budget_para_list = list(map(int, config['budget_para']))

    model_loss_cnt = 0
    reward_loss_cnt = 0
    train_reward_cnt = 0
    test_reward_cnt = 0

    for i in budget_para_list:
        RL = DQN(
            i,
            config['campaign_id'],
            config['feature_num'],
            config['action_num'],
            [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08],
            config['lr'],
            config['memory_size'],
            config['batch_size'],
            config['replace_target_iter'],
            seed=1
        )

        print('当前预算条件{}'.format(i))
        main(i, RL, config)
