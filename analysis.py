import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *
import pandas as pd
from sklearn.manifold import TSNE
import pickle

def data_read(paths=[]):
    mine_values = []
    num_trajs = len(paths)
    mine_paths = paths
    csv_datas = []
    for p in mine_paths:
        with open(p,'rb') as f:
            csv_data = pickle.load(f)
        csv_datas.append(csv_data)
    return csv_datas


def list_to_np(data_list):
    np_array = np.zeros([len(data_list),len(str)])
    for i in range(len(data_list)):
        print(data_list[i])
        str = data_list[i][1:-1].split('    ')
        for j in range(len(str)):
            print(str[j])
            np_array[i,j]=float(str[j])
    return np_array

if __name__=="__main__":
    paths = []
    path = '2021_08_16_12_01_54'
    path = '2021_08_17_12_27_23'
    for i in range(1100000,1200000,5000):
        paths.append('/data2/zj/PEARL-Continual/outputpearl/metaworld/'+path+'/eval_trajectories/epoch%d.pkl'%(i))
    csv_datas = data_read(paths)
    task_encodings = np.zeros([10*20,64])
    tsne = TSNE()
    plt.figure()

    for i in range(10):
        for j in range(20):
            task_encodings[j+i*20] = csv_datas[j]['z_mean_task_%d'%i]
            print()

    color_list = ['red','blue','yellow','black','grey','pink','purple','green','gold','deeppink']
    # color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y', 'b', 'c']
    # color_list = [[0, 178, 238]]*10
    tsne_data = tsne.fit_transform(task_encodings)
    for j in range(3):
        plt.scatter(tsne_data[j*20:(j+1)*20, 0], tsne_data[j*20:(j+1)*20, 1],c=color_list[j])
        # print(tsne_data[j*1:(j+1)*1, 0], tsne_data[j*1:(j+1)*1, 1])
    for i in range(20):
        plt.text(tsne_data[i, 0], tsne_data[i, 1],i)
        # plt.text(tsne_data[j, 0], tsne_data[j, 1], j)
    plt.savefig('./analysis/imitation.png')



    # tsne_data = tsne.fit_transform(task_0_encodings)
    # print(tsne_data.shape)
    # for i in range(100):
    #     plt.scatter(tsne_data[i,0],tsne_data[i,1])
    #     plt.text(tsne_data[i,0],tsne_data[i,1],i)
    # plt.show()



