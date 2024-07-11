'''testing the pdcoea algorithm on the bilinear function'''

import pdcoea
import numpy as np

def bilinear(x,y,alpha,beta):
    return (sum(y)*(sum(x)-(beta*len(y))))-(alpha*len(y)*sum(x))


print(pdcoea.pdcoea(pop_size=10,
                    chromosome_len=1000,
                    epochs=10000,
                    g=bilinear,
                    alpha=0.7,
                    beta=0.3))


