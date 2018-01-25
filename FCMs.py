# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 14:53:01 2016

@author: shanchao yang
"""
import numpy as np
import matplotlib.pyplot as plt


# 产生长度为 K 的 sequence
def generate_sequence(init_vec, weight, K, sequence):
    # 复制init vector 到 sequence第一列
    Nc = len(sequence)
    for i in range(Nc):
        sequence[i][0] = init_vec[i]
    # 迭代计算后面
    for k in range(1, K):
        for node in range(Nc):
            # if node == Nc - 2:
            #     print('find')
            sumV = 0
            for j in range(Nc):
                # if j == node:
                #     continue
                sumV += weight[node, j] * (sequence[j, k - 1])
            sequence[node][k] = transferFunc(sumV, 5)


# 根据上一时刻的 concept value 计算下一时刻 的 concept value, sequence=[[old], [new]]
def getNewConceptVal(weight, sequence):
    Nc = len(weight)
    for node in range(Nc):
        sumV = sequence[node, 0]
        for j in range(Nc):
            sumV += weight[node, j] * (sequence[j, 0])
        sequence[node, 1] = transferFunc(sumV)


# 传递函数，暂定为 tanh
def transferFunc(x, belta=1, flag='-01'):  # todo: 换成其他函数
    if flag == '-01':
        return np.tanh(x)
    else:
        return 1 / (1 + np.exp(-belta * x))


def reverseFunc(y, belta=1, flag='-01'):
    if flag == '-01':
        if y > 0.99999:
            y = 0.99999
        elif y < -0.99999:
            y = -0.99999
        return np.arctanh(y)
    else:
        if y > 0.999:
            y = 0.999

        elif y < 0.00001:
            y = 0.001
        # elif -0.00001 < y < 0:
        #     y = -0.00001

        x = 1 / belta * np.log(y / (1 - y))
        return x


def draw_seq(sequence):
    Nc = len(sequence)
    for i in range(Nc):
        plt.plot(sequence[i, :])
    plt.xlabel('state number')
    plt.ylabel('value of node')


def generate_sparse_w(weight, Nc, density=0.4, flag=0):
    # generate weight matrix with given density
    # flag = 1: each node has the same amount of neighbors, say int(Nc * density)
    # flag = 0: each node will  have a neighbor if  rand() < density.
    if flag == 0:
        for i in range(Nc):
            for j in range(Nc):
                if np.random.rand() < density:
                    weight[i, j] = 2 * np.random.rand() - 1
                    while np.abs(weight[i, j]) < 0.05:
                        weight[i, j] = 2 * np.random.rand() - 1
    else:
        nNeighbor = np.floor(Nc * density)
        for i in range(Nc):
            sample_index = np.random.choice(range(Nc), nNeighbor)
            for j in sample_index:
                    weight[i, j] = 2 * np.random.rand() - 1
                    while np.abs(weight[i, j]) < 0.05:
                        weight[i, j] = 2 * np.random.rand() - 1





def calmatrixError(W1, W2):
    err = 0
    ssMean = 0
    Nc = len(W1)

    TP = 0
    FN = 0
    TN = 0
    FP = 0

    for j in range(Nc):
        for i in range(Nc):
            # calculate the model_Error and SS mean
            if abs(W1[j, i]) < 0.05 < abs(W2[j, i]):
                err += np.abs(W2[j, i])
                FN += 1
            elif abs(W1[j, i]) > 0.05 > abs(W2[j, i]):
                err += abs(W1[j, i])
                FP += 1

            elif abs(W1[j, i]) > 0.05 and abs(W2[j, i]) > 0.05:
                err = err + abs(W1[j, i] - W2[j, i])
                TN += 1
            elif abs(W1[j, i]) < 0.05 and abs(W2[j, i]) < 0.05:
                TP += 1
            else:
                pass

    modelErr = err / Nc / Nc
    if TP + FN > 0:
        specificity = TP / (TP + FN)
    else:
        specificity = 1

    if TN + FP > 0:
        sensitivity = TN / (TN + FP)
    else:
        sensitivity = 1
    if TP + TN == 0:
        ssMean = 0
    else:
        ssMean = 2 * specificity * sensitivity / (specificity + sensitivity)

    return modelErr, ssMean


def objFunc(weight, trueSeq):
    # number of nodes
    Nc = len(trueSeq)
    # number of samples
    m = len(trueSeq[0, :])
    err = 0
    for node_i in range(1):
        # iterate samples
        for k in range(m - 1):
            net = transferFunc(np.dot(weight[node_i, :], trueSeq[:, k]))
            err += 1 / (2 * m) * np.power(trueSeq[node_i, k+1] - net, 2)

    return err


def gradientDecent(w1, w, seq, alpha,  maxIter=500):
    # number of nodes
    Nc = len(seq)
    # number of samples
    m = len(seq[0, :])
    # error function of each iteration
    J = np.zeros(shape=(maxIter, 1))
    bestW = np.zeros(shape=(Nc, Nc))
    bestJ = np.inf
    for Iter in range(maxIter):
        print("Iteration %d" % (Iter+1))
        # calculate new concept value A of step k

        # w_old = w.copy()
        # update the weights
        for node_i in range(1):
            # iterate samples
            sample_index = list(range(m-1))
            # np.random.shuffle(sample_index)
            for k in sample_index:
                # update each parameter
                for node_j in range(Nc):
                    if abs(w1[node_i, node_j]) < 0.05:
                        continue
                    # update formula
                    # w_ij  ->  in the codes,   node i to node j
                    h_x = np.dot(w[node_i, :], seq[:, k])
                    # linear regression
                    w[node_i, node_j] += alpha * (seq[node_i, k+1] - seq[node_j, k] * w[node_i, node_j] ) * seq[node_j, k]
                    # logistic regression
                    # w[node_i, node_j] += alpha * h_x * (1 - h_x) * (seq[node_i, k+1] - h_x) * seq[node_j, k]
                    # Hebbian rule
                    # w[node_i, node_j] += alpha * (seq[node_i, k+1] - seq[node_j, k] * w[node_i, node_j] ) * seq[node_j, k]

        J[Iter, 0] = objFunc(w, seq)
        if bestJ > J[Iter, 0]:
            bestJ = J[Iter]
            bestW = w.copy()
    return bestW, J


def main():
    # 专家给定的初始权重
    w_initial = np.array([[0, 0, 0.21, 0.48, 0, 0, 0, 0],
                          [0, 0, 0, 0.70, 0.60, 0, 0, 0],
                          [0.21, 0, 0, 0, 0, 0, 0, 0],
                          [-0.80, 0.7, 0, 0, 0, 0, 0.09, 0],
                          [0, -0.42, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0.4, 0, 0, 0, 0, 0.5],
                          [0, 0, 0, 0.3, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0.4, 0]], dtype=np.float32)
    # 初始状态
    state_initial = [0.48, 0.57, 0.58, 0.68, 0.59, 0.59, 0.52, 0.58]
    # 节点个数
    Nc = len(state_initial)
    # # 时间序列长度
    K = 12
    #
    seq = np.zeros(shape=(Nc, K), dtype=np.float32)
    generate_sequence(state_initial, w_initial, K, seq)
    # learning rate
    alpha = 1
    # errTest = objFunc(w_initial, seq)
    w0 = np.random.rand(Nc, Nc)
    w0[0, np.where(w_initial[0, :] == 0)[0]] = 0
    w_NHL, J = gradientDecent(w_initial, w0, seq, alpha)
    #
    print(w_NHL[0, :])
    print('minimum of J is %f' % min(J))
    plt.plot(J[:, 0])
    plt.show()

    # print(A_final[:, 1])
    # draw(seq)
    # plt.show()

if __name__ == '__main__':
    main()