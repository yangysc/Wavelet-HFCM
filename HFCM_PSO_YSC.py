import numpy as np
import matplotlib.pyplot as plt
from FCMs import transferFunc, reverseFunc
import pandas as pd
import time
import FuzzyCluster as fc
from sklearn import linear_model


def splitData(dataset, ratio=0.85):
    len_train_data = int(len(dataset) * ratio)
    return dataset[:len_train_data], dataset[len_train_data:]


def HFCM_ridge(dataset):
    # number of nodes
    Nc = 10
    # order of HFCMs
    Order = 4
    # steepness of sigmoid function
    belta = 1

    # partition dataset into train set and test set
    ratio = 0.7
    if len(dataset) > 2 * Order / (1 - ratio):
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data = dataset[:]
        test_data = dataset[:]
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    U_train, center = fc.fcm(train_data, Nc)

    # 30 independent runs

    import lightning.regression as rg
    alpha = 1e-2
    eta_svrg = 1e-2
    tol = 1e-24
    start = time.time()
    clf = rg.SVRGRegressor(alpha=alpha, eta=eta_svrg,
                           n_inner=1.0, max_iter=100, random_state=0, tol=tol)
    # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)
    # learned weight matrix
    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
    samples_train = {}
    for node_solved in range(Nc):  # solve each node in turn
        samples = create_dataset(U_train, belta, Order, node_solved)
        # delete last "Order" rows (all zeros)
        samples_train[node_solved] = samples[:-Order, :]
        clf.fit(samples[:, :-1], samples[:, -1])
        W_learned[node_solved, :] = clf.coef_

    steepness = np.max(np.abs(W_learned), axis=1)
    for i in range(Nc):
        if steepness[i] > 1:
            W_learned[i, :] /= steepness[i]

    # print(W_learned)
    # save U_train and W_learned into local files(in order to test the correctness of HFCM_PSO)
    # import pickle
    # file = open('U_train_and_W_learned', 'wr')
    # pickle.dump(U_train, file)
    # pickle.dump(W_learned, file)
    # pickle.dump(steepness, file)
    # file.close()
    # predict on training data set
    # trainPredict = np.zeros(shape=(Nc, len_train_data - Order))
    # for i in range(Nc):
    #     trainPredict[i, :] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(211)
    # # fig1.hold()
    # for i in range(Nc):
    #     ax1.plot(U_train[i, Order:])
    #
    # ax1.set_xlabel('n')
    # ax1.set_title('Membership of train data')
    # ax2 = fig1.add_subplot(212)
    # for i in range(Nc):
    #     ax2.plot(trainPredict[i, :])
    # ax2.set_xlabel('n')
    # ax2.set_title('Membership of predicted train data')
    # fig1.tight_layout()
    # plt.show(3)
    return U_train, W_learned, steepness


# form feature matrix from sequence
def create_dataset(seq, belta, Order, current_node):
    Nc, K = seq.shape
    samples = np.zeros(shape=(K, Order * Nc + 2))
    for m in range(Order, K):
        for n_idx in range(Nc):
            for order in range(Order):
                samples[m - Order, n_idx * Order + order + 1] = seq[n_idx, m - order]
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = reverseFunc(seq[current_node, m], belta)
    return samples


def predict(samples, weight, steepness, belta):
    # samples: each row is a sample, each column is one feature
    K, _ = samples.shape
    predicted_data = np.zeros(shape=(1, K))
    for t in range(K):
        features = samples[t, :-1]
        predicted_data[0, t] = steepness * np.dot(weight, features)
        predicted_data[0, t] = transferFunc(predicted_data[0, t], belta)
    return predicted_data


# Given weights and initial nodes' state, generate the whole sequence
def generate_seq(true_seq, weight, Nc, K, Order, beta):

    seq = np.zeros(shape=(Nc, K))
    # seq[:, :Order] = true_seq[:, :Order]
    # length of variables of each node(weights + bias + beta)
    len_var = Order * Nc + 1
    for sample in range(Order, K-1):  # iterate each sample
        for j in range(Nc):    # iterate each node j
            sumV = weight[j * len_var + Order * Nc]  # bias
            for k in range(Nc):  # iterate j's neibors
                for m in range(Order):
                    sumV += weight[j * len_var + k * Order + m] * true_seq[k][sample - 1 - m]
            seq[j][sample] = transferFunc(sumV, beta)
    return seq[:, Order:]


# calculate the fitness of each candidate
def obj_eval(seq, true_seq, Order):
    Nc, K = seq.shape
    err = 0
    for k in range(Order, K):
        for node in range(Nc):
            err += np.power(seq[node][k] - true_seq[node][k], 2)
    err /= (K - 1 - Order) * Nc
    return err


def objfunc_PSO(x, *args):
    Nc, K, Order, beta, true_seq = args
    seq = generate_seq(true_seq, x, Nc, K, Order, beta)
    err = obj_eval(seq, true_seq, Order)
    return err


# main function
def HFCM_PSO():
    dataset = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]
    minV = np.min(dataset)
    maxV = np.max(dataset)
    dataset = (dataset - minV) / (maxV - minV)
    # number of nodes
    Nc = 10
    # order of HFCMs
    Order = 4
    # steepness of sigmoid function
    beta = 1
    # number of variable of each node
    nvar = Order * Nc + 1
    # range of weights
    lowBnd = [0] * nvar
    uppBnd = [0] * nvar
    # range of weights
    for i in range(Order * Nc):
        lowBnd[i] = -1
        uppBnd[i] = 1
    # range of bias
    lowBnd[Order * Nc] = 0
    uppBnd[Order * Nc] = 1
    # # range of beta
    # lowBnd[Order * Nc + 1] = 1
    # uppBnd[Order * Nc + 1] = 2

    lb = np.tile(lowBnd, Nc)
    ub = np.tile(uppBnd, Nc)
    # partition dataset into train set and test set
    ratio = 0.7
    if len(dataset) > 2 * Order / (1 - ratio):
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data = dataset[:]
        test_data = dataset[:]
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    U_train, center = fc.fcm(train_data, Nc)

    # use PSO to learn weight matrix
    from pyswarm import pso
    true_seq = U_train
    args = (Nc, len_train_data, Order, beta, true_seq)
    # lb = [lowBund] * Nc
    # ub = [uppBnd] * Nc
    # for (i = 0; i < Order * Nc * Nc; i++){
    # lowBound[i] = -1;
    # uppBound[i] = 1;
    # }
    # // bias
    # for (; i < Order * Nc * Nc + Nc; i++)
    # {
    #     lowBound[i] = 0;
    # uppBound[i] = 1;
    # }
    # // steepness
    # parameter
    # for (; i < nvar; i++)
    # {
    #     lowBound[i] = 1;
    # uppBound[i] = 10;
    # }
    start = time.time()
    xopt, fopt = pso(objfunc_PSO, lb, ub, args=args, swarmsize=50, omega=0.9, phig=2.1, phip=2.1, maxiter=500)
    endT = time.time()
    print('PSO use %f (s)' % (endT - start))
    print('x is')
    print(xopt)
    print('min obj is %f' % fopt)
    trainPredict = generate_seq(true_seq, xopt, Nc, len_train_data, Order, beta)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    # fig1.hold()
    for i in range(Nc):
        ax1.plot(U_train[i, Order:])

    ax1.set_xlabel('n')
    ax1.set_title('Membership of train data')
    ax2 = fig1.add_subplot(212)
    for i in range(Nc):
        ax2.plot(trainPredict[i, :])
    ax2.set_xlabel('n')
    ax2.set_title('Membership of predicted train data')
    fig1.tight_layout()
    plt.show()


if __name__ == '__main__':

    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    # dataset = TAIEX.parse('clean_v1_2000')
    # main()
    HFCM_PSO()
    # debug HFCM_PSO
    # dataset = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]
    # minV = np.min(dataset)
    # maxV = np.max(dataset)
    # dataset = (dataset - minV) / (maxV - minV)
    # # number of nodes
    # Nc = 10
    # # order of HFCMs
    # Order = 4
    # # steepness of sigmoid function
    # belta = 1
    # # number of variable of each node
    # nvar = Order * Nc + 2
    #
    # ratio = 0.7
    # if len(dataset) > 2 * Order / (1 - ratio):
    #     train_data, test_data = splitData(dataset, ratio)
    # else:
    #     train_data = dataset[:]
    #     test_data = dataset[:]
    # len_train_data = len(train_data)
    # len_test_data = len(test_data)
    #
    # U_train, W_learned, steepness = HFCM_ridge(dataset)
    #
    # # range of weights
    # xopt = [0] * nvar * Nc
    # for i in range(Nc):
    #     xopt[i * nvar + Order * Nc] = W_learned[i, 0]  # bias
    #     xopt[i * nvar + Order * Nc + 1] = steepness[i]    # steepness
    #     xopt[i * nvar: i*nvar + Order * Nc] = W_learned[i, 1:]  # weights
    # print(xopt)
    # # print(steepness)
    # true_seq = U_train
    # trainPredict = generate_seq(true_seq[:, :Order], xopt, Nc, len_train_data, Order, true_seq)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(211)
    # # fig1.hold()
    # for i in range(Nc):
    #     ax1.plot(U_train[i, Order:])
    #
    # ax1.set_xlabel('n')
    # ax1.set_title('Membership of train data')
    # ax2 = fig1.add_subplot(212)
    # for i in range(Nc):
    #     ax2.plot(trainPredict[i, :])
    # ax2.set_xlabel('n')
    # ax2.set_title('Membership of predicted train data')
    # fig1.tight_layout()
    # plt.show()
    #
