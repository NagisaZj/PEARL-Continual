import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from plot_utils import *




def plot_full(data,color,name):
    plt.plot(data[0], data[1], color,label=name)
    plt.fill_between(data[0], data[1] - data[2], data[1] + data[2], color=color, alpha=0.1, linewidth=0)
    plt.plot(data[0], np.ones(data[0].shape) * np.mean(data[1][-20:]), color=color, linestyle=':')


def smoothingaverage(data,window_size=5):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data,window,'valid')

if __name__ =="__main__":
    mine_data=data_read(paths=['./outputfin2/reacher-goal-sparse/2019_12_03_18_35_30/progress.csv',
                  './outputfin2/reacher-goal-sparse/2019_12_03_18_35_35/progress.csv',
                  './outputfin2/reacher-goal-sparse/2019_12_03_10_02_31/progress.csv'])
    mine_data_new = data_read(paths=['./outputfin2/reacher-goal-sparse/new-intr1/progress.csv',
                                 './outputfin2/reacher-goal-sparse/new-intr2/progress.csv',
                                 './outputfin2/reacher-goal-sparse/new-intr3/progress.csv'])
    mine_data_new_intr = data_read(paths=['/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_43/progress.csv',
                                     '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_55/progress.csv',
                                     '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_35_03/progress.csv',
                                    '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_04_09_01_59/progress.csv',
                                     '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_04_09_01_48/progress.csv',
                                    '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_26_15_03_23/progress.csv',
                                    '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_26_15_03_18/progress.csv',])

    mine_data_new_intr = data_read(paths=[
        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_43/progress.csv',
        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_55/progress.csv',
        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_35_03/progress.csv',

        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_04_09_01_48/progress.csv',
        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_26_15_03_23/progress.csv',
        '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_12_26_15_03_18/progress.csv', ])

    pearl_data = data_read(paths=['./output/reacher-goal-sparse/2019_12_02_18_54_27/progress.csv',
                                  './output/reacher-goal-sparse/2019_12_02_18_54_35/progress.csv',
                                  './output/reacher-goal-sparse/2019_12_03_10_02_36/progress.csv'])
    pearl_data = data_read(paths=['./output/reacher-goal-sparse/2019_12_02_18_54_35/progress.csv',
                                  './output/reacher-goal-sparse/2019_12_02_18_54_27/progress.csv',
                                  './output/reacher-goal-sparse/2019_12_02_18_54_27/progress.csv'])
    pearl_data = data_read(paths=['/home/lthpc/Desktop/Research/temp/new-pearl/output/reacher-goal-sparse/2020_11_14_10_39_56/progress.csv',
                                  '/home/lthpc/Desktop/Research/temp/new-pearl/output/reacher-goal-sparse/2020_11_14_10_40_03/progress.csv',
                                  '/home/lthpc/Desktop/Research/temp/new-pearl/output/reacher-goal-sparse/2020_11_14_10_40_09/progress.csv',
                                  '/home/lthpc/Desktop/Research/temp/new-pearl/output/reacher-goal-sparse/2020_12_06_20_49_15/progress.csv',
                                  '/home/lthpc/Desktop/Research/temp/new-pearl/output/reacher-goal-sparse/2020_12_06_20_49_21/progress.csv'
                                  ])
    maml_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/maml/reacher-goal/new2/run_1575527921/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/maml/reacher-goal/new2/run_1575527925/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/maml/reacher-goal/new2/run_1575527929/progress.csv'])
    promp_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/pro-mp/reacher-goal/new/run_1575508965/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/pro-mp/reacher-goal/new/run_1575508968/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/pro-mp/reacher-goal/new/run_1575508971/progress.csv'])
    rl2_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/test_4881960120/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/test_583694845/progress.csv',
                                  '/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/test_9623184471/progress.csv'])

    erl2_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/1/progress.csv',
                                '/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/2/progress.csv',
                                '/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/3/progress.csv'])
    rl2_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/n1/progress.csv',
                                '/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/n2/progress.csv',
                                '/home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal/n3/progress.csv'])

    erl2_data = data_read(paths=['/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/n1/progress.csv',
                                 '/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/n2/progress.csv',
                                 '/home/lthpc/Desktop/Research/ProMP/data/erl2/reacher-goal/n3/progress.csv'])
    mame_data = data_read_mame(paths=['./outputmame/reacher/1',
                                      './outputmame/reacher/2'])
    varibad_data = data_read_varibad(paths=['./outputvaribad/reacher/1/returns.npy',
                                      './outputvaribad/reacher/2/returns.npy',
                                            './outputvaribad/reacher/2/returns.npy'])
    Maesn_data = data_read_Maesn(paths=[
        './outputmaesn/reacher-goal/1/progress.csv',
        './outputmaesn/reacher-goal/2/progress.csv',
        './outputmaesn/reacher-goal/3/progress.csv'])
    mine_data_new_2 = data_read(paths=['/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_43/progress.csv',
                                     '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_34_55/progress.csv',
                                     '/home/lthpc/Desktop/Research/temp/new-pearl/outputfin2/reacher-goal-sparse/2020_11_13_09_35_03/progress.csv'])

    #print(maml_data[0][-1],maml_data[1][-1])

    datas = [mine_data_new_intr, pearl_data, maml_data, varibad_data, rl2_data]
    legends = ['MetaCURE', 'PEARL', 'MAML', 'VariBAD', 'RL^2']
    #datas = [mine_data_new_intr, promp_data, erl2_data, mame_data]
    #legends = ['MetaCURE', 'ProMP', 'E-RL^2', 'MAME']
    plot_all(datas, legends, 0)
    plt.title('Reacher-Goal-Sparse', size=30)
    #plt.plot(mine_data_new_intr[0], np.ones(mine_data_new_intr[0].shape) * 1.31, color='olive', linestyle='--', linewidth=2, label='EPI')
    #legend()
    plt.show()



    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.title('Reacher-Goal-Sparse', size=30)
    plt.xlabel('Million Environment Samples', size=25)
    plt.ylabel('Average Return', size=25)

    plot_full(mine_data, 'r', 'EIMUR')
    plot_full(mine_data_new, 'mediumblue', 'EIMUR')
    plot_full(mine_data_new_2, 'purple', 'EIMUR')
    plot_full(pearl_data, 'forestgreen', 'PEARL')
    #plot_full(maml_data, 'purple', 'MAML')
    #plot_full(varibad_data, 'deeppink', 'VariBAD')
    #plot_full(rl2_data, 'c', 'RL^2')
    plot_full(promp_data, 'y', 'ProMP')
    plot_full(erl2_data, 'coral', 'E-RL^2')
    plot_full(mame_data, 'brown', 'MAME')

    #plot_full(Maesn_data, 'olive', 'MAESN')
    plt.plot(mine_data[0], np.ones(mine_data[0].shape) * 1.31, color='k', linestyle='--',linewidth=2)



    #plt.plot(pearl_data[0], pearl_data[1], 'b')
    #plt.fill_between(pearl_data[0], pearl_data[1] - pearl_data[2], pearl_data[1] + pearl_data[2], color='b', alpha=0.2)
    #plt.plot(pearl_data[0], np.ones(pearl_data[0].shape) * np.max(pearl_data[1]), 'b:')
    #plt.legend()
