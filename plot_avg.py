import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp


WINDOW_LENGTH = 20
SMOOTH_COEF = 0.05
CM = 1 / 2.54  # centimeters in inches


CURVE_FORMAT = {
    'sgd': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'SGD',
    },
'2021_08_13_11_27_06': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'SGD',
    },
'2021_08_16_03_42_07': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'PEARL imitation',
    },
'2021_08_16_12_01_54': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'PEARL continual',
    },
    'ewc_lambda100': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'ewc_lambda100',
    },
    'ewc_lambda200': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'ewc_lambda200',
    },
    'ewc_lambda500': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'ewc_lambda500',
    },
    'ewc_lambda1000': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'ewc_lambda1000',
    },
    'ewc_lambda2000': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'ewc_lambda2000'
    },
    'ewc_lambda5000': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'ewc_lambda5000'
    },
    'ewc_lambda8000': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'EWC_lambda8000'
    },
    'ewc_lambda10000': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'ewc_lambda10000'
    },
    'ewc_lambda20000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'ewc_lambda20000'
    },
    'si_c0.1': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'si_c0.1',
    },
    'si_c1': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'si_c1',
    },
    'si_c5': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'si_c5',
    },
    'si_c10': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'si_c10',
    },
    'si_c25': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'si_c25'
    },
    'si_c50': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'si_c50'
    },
    'si_c100': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'si_c100'
    },
    'si_c200': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'si_c200'
    },
    'si_c500': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'si_c500'
    },
    'agem_ref_grad_batch_size250': {
        'color': [0, 178, 238],
        'style': '-',
        'label': 'agem_ref_grad_batch_size250',
    },
    'agem_ref_grad_batch_size500': {
        'color': [204, 153, 255],
        'style': '-',
        'label': 'agem_ref_grad_batch_size500',
    },
    'agem_ref_grad_batch_size1000': {
        'color': [139, 101, 8],
        'style': '-',
        'label': 'agem_ref_grad_batch_size1000',
    },
    'agem_ref_grad_batch_size2500': {
        'color': [0, 100, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size2500',
    },
    'agem_ref_grad_batch_size4500': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size4500'
    },
    'agem_ref_grad_batch_size5000': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size5000'
    },
'agem_ref_grad_batch_size5000_noobs': {
        'color': [255, 128, 0],
        'style': '-',
        'label': 'agem_ref_grad_batch_size5000_noobs'
    },
    'agem_ref_grad_batch_size7500': {
        'color': [160, 32, 240],
        'style': '-',
        'label': 'agem_ref_grad_batch_size7500'
    },
    'agem_ref_grad_batch_size10000': {
        'color': [216, 30, 54],
        'style': '-',
        'label': 'agem_ref_grad_batch_size10000'
    },
    'agem_ref_grad_batch_size15000': {
        'color': [55, 126, 184],
        'style': '-',
        'label': 'agem_ref_grad_batch_size15000'
    },
    'agem_ref_grad_batch_size20000': {
        'color': [180, 180, 180],
        'style': '-',
        'label': 'agem_ref_grad_batch_size20000'
    },
}


def window_smooth(y):
    window_size = int(WINDOW_LENGTH / 2)
    y_padding = np.concatenate([y[:1] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y_padding, y], axis=0)
    y_padding = np.concatenate([y[-1:] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y, y_padding], axis=0)

    coef = list()
    for i in range(WINDOW_LENGTH + 1):
        coef.append(np.exp(- SMOOTH_COEF * abs(i - window_size)))
    coef = np.array(coef)

    yw = list()
    for t in range(len(y)-WINDOW_LENGTH):
        yw.append(np.sum(y[t:t+WINDOW_LENGTH+1] * coef) / np.sum(coef))

    return np.array(yw).flatten()


def plot(ax, data, task_names, algos, curve_format=CURVE_FORMAT):
    algo_norm_data = {}
    for task_name, oracle_return in task_names:
        oracle_return = float(oracle_return)

        for algo in algos:
            algo_data = data[task_name][algo]
            if 'y' not in algo_data:
                continue

            # if len(algo_data['x']) != len(algo_data['y'][0]):
            #     min_len = len(algo_data['x'])
            #     for y in algo_data['y']:
            #         min_len = min(min_len, len(y))
            #
            #     algo_data['x'] = algo_data['x'][:min_len]
            #     for idx, y in enumerate(algo_data['y']):
            #         algo_data['y'][idx] = y[:min_len]
            y_len = 1E10
            for y in algo_data['y']:
                y_len = min(len(y), y_len)

            algo_data['x'] = algo_data['x'][:y_len]
            for idx in range(len(algo_data['y'])):
                algo_data['y'][idx] = algo_data['y'][idx][:y_len]

            # normalize via oracle returns
            y_mean = np.mean(np.array(algo_data['y']) / oracle_return, axis=0)
            if algo not in algo_norm_data:
                algo_norm_data[algo] = {}
                algo_norm_data[algo]['x'] = algo_data['x']
                algo_norm_data[algo]['y'] = [y_mean]
            else:
                algo_norm_data[algo]['y'].append(y_mean)

    for algo in algos:
        if algo not in algo_norm_data:
            continue

        algo_data = algo_norm_data[algo]

        if 'y' not in algo_data:
            continue

        y_len = 1E10
        for y in algo_data['y']:
            y_len = min(len(y), y_len)

        x = algo_data['x'][:y_len]
        for idx in range(len(algo_data['y'])):
            algo_data['y'][idx] = algo_data['y'][idx][:y_len]
        print(np.array(algo_data['y']).shape)
        y_mean = np.mean(np.array(algo_data['y']), axis=0)
        y_std = np.std(np.array(algo_data['y']), axis=0)
        y_mean = window_smooth(y_mean)
        y_std = window_smooth(y_std)

        color = np.array(curve_format[algo]['color']) / 255.
        style = curve_format[algo]['style']
        label = curve_format[algo]['label']
        ax.plot(x, y_mean, color=color, label=label, linestyle=style)
        ax.fill_between(x, y_mean - 0.5 * y_std, y_mean + 0.5 * y_std, facecolor=color, alpha=0.1)


def main(args):
    # exp_name = args.exp_name + '-setting-' + str(args.setting)
    exp_name = args.exp_name
    task_names = args.task_names
    algos = args.algos
    data_dir = args.data_dir
    save_dir = args.save_dir
    stats = args.statistics
    seeds = args.seeds
    max_timesteps = args.max_timesteps
    # num_fig = len(stats)

    os.makedirs(save_dir, exist_ok=True)

    fig, _ = plt.subplots(1, len(stats))
    fig.set_size_inches(10 * len(stats), 8)
    for stat_idx, stat in enumerate(stats):
        ax = plt.subplot(1, len(stats), stat_idx + 1)
        # ax.set_title(task_name, fontsize=15)
        ax.set_xlabel('Total Timesteps', fontsize=15)
        ax.set_ylabel('normalized_' + stat, fontsize=15)
        data = {}
        i = 0
        for task_name, _ in task_names:
            data[task_name] = {}
            for algo in algos:
                data[task_name][algo] = {}
                data_path = '/data2/zj/PEARL-Continual/outputpearl/metaworld/'+algo+'/progress.csv'


                try:
                    df = pd.read_csv(data_path)
                except:
                    print(f"Data path not found: {data_path}!")
                    continue

                # file = file[file['step'] <= args.timesteps]
                # x = file['exploration/all num steps total'].values

                task_df = df['AverageReturn_task_last_%d'%i]
                print(i,task_name)
                # task_df = task_df[task_df['step'] <= max_timesteps]
                data[task_name][algo].update(x=df['Number of env steps total'].values)

                try:
                    y = task_df.values
                    if 'y' not in data[task_name][algo]:
                        data[task_name][algo].update(y=[y])
                    else:
                        data[task_name][algo]['y'].append(y)
                except:
                    raise RuntimeError(f"Statistics '{stat}' doesn't exist in '{data_path}'!")
            i += 1

        plot(ax, data, task_names, algos)

    fig_path = osp.abspath(osp.join(save_dir, exp_name + '.png'))
    # plt.title(exp_name, fontsize=16)
    fig.suptitle(exp_name, fontsize=20).set_y(0.9875)
    plt.tight_layout()
    plt.legend(framealpha=0.)
    plt.savefig(fname=fig_path)
    print(f"Save figure: {fig_path}")


if __name__ == '__main__':
    # custom argument type
    def str_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return (splited_s[0], splited_s[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='reach_window-close_button-press-topdown')
    parser.add_argument('--data_dir', type=str, default='vec_logs')
    parser.add_argument('--save_dir', type=str, default='figures_avg')
    # parser.add_argument('--setting', type=int, default=1)
    parser.add_argument('--task_names', type=str_pair, nargs='+', default=[
        ('window-close-v2', '4580.912'),
        ('button-press-topdown-v2', '3894.807'),
        ('door-open-v2', '2958.152'),
        ('coffee-button-v2', '3992.783'),
        ('plate-slide-side-v2', '4324.899'),
        ('push-v2', '3194.469'),
        ('faucet-close-v2', '3194.469'),
        ('handle-press-v2', '4504.751'),
        ('reach-wall-v2', '4844.363'),
        ('window-open-v2', '4240.124')
    ])

    parser.add_argument('--algos', type=str, nargs='+',
                        default=['sgd', 'ewc', 'si'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[ 4, 5, 6, 7])
    parser.add_argument('--max_timesteps', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('--statistics', type=str, nargs='+',
                        default=['episode_reward'])
    args = parser.parse_args()

    main(args)
