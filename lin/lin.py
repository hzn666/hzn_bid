import argparse
import os

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def get_budget(data):
    _ = []
    for day in data['day'].unique():
        current_day_budget = sum(data[data['day'].isin([day])]['market_price'])
        _.append(current_day_budget)

    return _


def bid(data, budget, config):
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

    data['bid_price'] = np.clip(
        np.multiply(np.divide(data['pctr'], config['average_pctr']), config['base_bid']).astype(int), 0, 300)
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


def rtb(data, budget_para, config, train=True):
    budget = get_budget(data)
    budget = np.divide(budget, budget_para)

    bid_imps = []
    bid_clks = []
    bid_pctr = []
    win_imps = []
    win_clks = []
    win_pctr = []
    spend = []
    bid_action = []
    day_result = []

    for day_index, day in enumerate(data['day'].unique()):
        day_data = data[data['day'].isin([day])]
        day_budget = budget[day_index]

        bid_result = bid(day_data, day_budget, config)

        bid_imps.append(bid_result['bid_imps'])
        bid_clks.append(bid_result['bid_clks'])
        bid_pctr.append(bid_result['bid_pctr'])
        win_imps.append(bid_result['win_imps'])
        win_clks.append(bid_result['win_clks'])
        win_pctr.append(bid_result['win_pctr'])
        spend.append(bid_result['spend'])
        bid_action.extend(bid_result['bid_action'])

        day_result.append([
            budget_para,
            config['base_bid'],
            day,
            win_clks[-1],
            bid_clks[-1],
            win_pctr[-1],
            bid_pctr[-1],
            win_imps[-1],
            bid_imps[-1],
            spend[-1]
        ])

    print("预算条件 {}, 基础出价 {}, 点击数 {}, 真实点击数 {}, pCTR {:.4f}, 真实pCTR {:.4f}, 赢标数 {}, 真实曝光数 {}, 花费 {},".format(
        budget_para,
        config['base_bid'],
        sum(win_clks),
        sum(bid_clks),
        sum(win_pctr),
        sum(bid_pctr),
        sum(win_imps),
        sum(bid_imps),
        sum(spend)
    ))

    result = [
        budget_para,
        config['base_bid'],
        sum(win_clks),
        sum(bid_clks),
        sum(win_pctr),
        sum(bid_pctr),
        sum(win_imps),
        sum(bid_imps),
        sum(spend),
        config['average_pctr']
    ]

    if train:
        return result, day_result
    else:
        return result, day_result, bid_action


def main(config):
    train_data = pd.read_csv(config['train_data_path'])
    test_data = pd.read_csv(config['test_data_path'])

    header = ['clk', 'pctr', 'market_price', 'day', '24_time_fraction']
    train_data = train_data[header]
    train_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']
    test_data = test_data[header]
    test_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']

    average_pctr = np.sum(train_data.pctr) / len(train_data)
    config['average_pctr'] = average_pctr

    train_result = []
    train_day_result = []

    budget_para_list = list(map(int, config['budget_para']))
    for budget_para in budget_para_list:
        print(config['train_type'] + "_训练_" + str(budget_para))
        for base_bid in np.arange(1, 301):
            config['base_bid'] = base_bid
            result, day_result = rtb(train_data, budget_para, config)
            train_result.append(result)
            train_day_result.extend(day_result)

    print("存储全体训练日志")
    train_log = pd.DataFrame(
        data=train_result,
        columns=[
            'budget_para', 'base_bid', 'win_clks', 'real_clks', 'win_pctr', 'real_pctr', 'win_imps', 'real_imps',
            'spend', 'average_pctr'
        ]
    )
    train_log.to_csv(os.path.join(config['train_log_path'], 'train_bid_log.csv'), index=False)

    print("存储单天训练日志")
    day_train_log = pd.DataFrame(
        data=train_day_result,
        columns=[
            'budget_para', 'base_bid', 'day', 'win_clks', 'real_clks', 'win_pctr', 'real_pctr', 'win_imps', 'real_imps',
            'spend'
        ]
    )
    day_train_log.to_csv(os.path.join(config['train_log_path'], 'day_train_bid_log.csv'), index=False)

    print("存储全体训练最优出价")
    clk_best_base_bid = train_log.groupby(['budget_para']).apply(lambda x: x[x.win_clks == x.win_clks.max()])
    clk_best_base_bid.to_csv(os.path.join(config['train_log_path'], 'train_clk_best_base_bid.csv'), index=False)
    pctr_best_base_bid = train_log.groupby(['budget_para']).apply(lambda x: x[x.win_pctr == x.win_pctr.max()])
    pctr_best_base_bid.to_csv(os.path.join(config['train_log_path'], 'train_pctr_best_base_bid.csv'), index=False)

    print("存储单天训练最优出价")
    for day in train_data.day.unique():
        current_day_data = day_train_log[day_train_log.day.isin([day])]

        clk_day_best_base_bid = current_day_data.groupby(['budget_para']).apply(
            lambda x: x[x.win_clks == x.win_clks.max()])
        clk_day_best_base_bid.to_csv(
            os.path.join(config['train_log_path'], 'clk_day{}_train_best_base_bid.csv'.format(day)),
            index=False)
        pctr_day_best_base_bid = current_day_data.groupby(['budget_para']).apply(
            lambda x: x[x.win_pctr == x.win_pctr.max()])
        pctr_day_best_base_bid.to_csv(
            os.path.join(config['train_log_path'], 'pctr_day{}_train_best_base_bid.csv'.format(day)),
            index=False)

    if config['metrics'] == 'clk':
        best_base_bid_df = pd.read_csv(os.path.join(config['train_log_path'], 'train_clk_best_base_bid.csv'))
    else:
        best_base_bid_df = pd.read_csv(os.path.join(config['train_log_path'], 'train_pctr_best_base_bid.csv'))

    test_result = []
    test_day_result = []

    for budget_para in budget_para_list:
        print(config['train_type'] + "_测试_" + str(budget_para))

        best_base_bid_row = best_base_bid_df[best_base_bid_df.budget_para.isin([budget_para])]
        best_base_bid = best_base_bid_row.iloc[0].base_bid
        config['base_bid'] = best_base_bid
        result, day_result, bid_action = rtb(test_data, budget_para, config, train=False)
        test_result.append(result)
        test_day_result.extend(day_result)

        print("存储不同预算条件下测试集的出价动作")
        bid_action_df = pd.DataFrame(
            data=bid_action,
            columns=[
                'clk', 'pctr', 'market_price', 'day', 'time_fraction', 'bid_price', 'win'
            ]
        )
        bid_action_df.to_csv(os.path.join(config['test_log_path'], '{}_test_bid_action.csv'.format(budget_para)),
                             index=False)

    print("存储全体测试日志")
    test_log = pd.DataFrame(
        data=test_result,
        columns=[
            'budget_para', 'base_bid', 'win_clks', 'real_clks', 'win_pctr', 'real_pctr', 'win_imps', 'real_imps',
            'spend', 'average_pctr'
        ]
    )
    test_log.to_csv(os.path.join(config['test_log_path'], 'test_bid_log.csv'), index=False)

    print("存储单天测试日志")
    day_test_log = pd.DataFrame(
        data=test_day_result,
        columns=[
            'budget_para', 'base_bid', 'day', 'win_clks', 'real_clks', 'win_pctr', 'real_pctr', 'win_imps', 'real_imps',
            'spend'
        ]
    )
    day_test_log.to_csv(os.path.join(config['test_log_path'], 'day_test_bid_log.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458')
    parser.add_argument('--result_path', type=str, default='result/ipinyou')
    parser.add_argument('--budget_para', nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--train_type', default='normal', help='normal or reverse')
    parser.add_argument('--metrics', default='clk', help='clk or pctr')

    args = parser.parse_args()
    config = vars(args)

    if config['train_type'] == 'reverse':
        config['train_data'] = 'test.bid.lin.csv'
        config['test_data'] = 'train.bid.lin.csv'
    else:
        config['train_data'] = 'train.bid.lin.csv'
        config['test_data'] = 'test.bid.lin.csv'

    camp = config['campaign_id']
    config['train_data_path'] = os.path.join(config['data_path'], camp, config['train_data'])
    config['test_data_path'] = os.path.join(config['data_path'], camp, config['test_data'])

    config['result_path'] = os.path.join(config['result_path'], camp, config['train_type'])
    config['train_log_path'] = os.path.join(config['result_path'], 'train')
    config['test_log_path'] = os.path.join(config['result_path'], 'test')

    if not os.path.exists(config['train_log_path']):
        os.makedirs(config['train_log_path'])
    if not os.path.exists(config['test_log_path']):
        os.makedirs(config['test_log_path'])

    main(config)
