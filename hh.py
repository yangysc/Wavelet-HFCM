import imfun
import numpy as np
Order = 4
a = imfun.atrous.decompose(np.array(range(100)), 3)
re_a = imfun.atrous.rec_atrous(a[:, Order])
# print a
print(re_a)