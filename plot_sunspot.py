import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == '__main__':

    y = []
    with open('sunspot.csv', 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            # try:
            #     # print(row[0], end=' ')
            #     # print(row[1])
            # except IndexError as e:
            #     print(row)
            #     exit(-4)

            y.append(float(row[1]))

    print(y)
    print(len(y))
    delta_y = []
    for i in range(1, len(y)):
        delta_y.append(y[i] - y[i-1])
    # # plt.close(2)
    # plt.clf()
    # plt.plot(range(len(y)), y, '-o')
    # plt.xlabel('time /month')
    # plt.ylabel('sunspot numbers')
    # plt.title('zuerich monthly sunspot numbers')
    #
    # plt.figure(2)
    # plt.plot(range(len(delta_y)), delta_y, '-o')
    # plt.xlabel('time /month')
    # plt.ylabel('difference of sunspot numbers')
    # plt.title('difference of zuerich monthly sunspot numbers')

    plt.figure(3)
    plt.plot(y[1:], delta_y, 'o', c='grey')
    plt.hold()
    V = np.loadtxt('fuzzyV.txt')

    for i in range(len(V)):
        plt.scatter(V[i, 0], V[i, 1])
    plt.xlabel('sunspot numbers')
    plt.ylabel('difference of sunspot numbers')
    plt.title('relation between of x and y')
    plt.show()