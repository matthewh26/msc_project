'''testing the pdcoea algorithm on the bilinear function'''

import pdcoea
import numpy as np
import matplotlib.pyplot as plt

def bilinear(x,y,alpha,beta):
    return (sum(y)*(sum(x)-(beta*len(y))))-(alpha*len(y)*sum(x))


print(pdcoea.pdcoea(pop_size=10,
                    chromosome_len=100,
                    epochs=1000,
                    g=bilinear,
                    sample=True,
                    alpha=0.7,
                    beta=0.3))


