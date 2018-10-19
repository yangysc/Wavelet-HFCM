# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 14:53:01 2016

@author: shanchao yang
"""
import numpy as np
import matplotlib.pyplot as plt




# the trasfer function of FCMS,  tanh is used here.
def transferFunc(x, belta=1, flag='-01'):
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
