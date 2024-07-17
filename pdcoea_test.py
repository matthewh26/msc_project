'''testing the pdcoea algorithm on the bilinear function'''

import pdcoea
import numpy as np
import matplotlib.pyplot as plt

def bilinear(x,y,alpha,beta):
    return (sum(y)*(sum(x)-(beta*len(y))))-(alpha*len(y)*sum(x))


a,b,totals = pdcoea.pdcoea(pop_size=10,
                    chromosome_len=100,
                    epochs=500,
                    g=bilinear,
                    sample=True,
                    alpha=0.7,
                    beta=0.3)

a_totals = [x[0] for x in totals]
b_totals = [x[1] for x in totals]


#plot for the test on bilinear the average magnitude of the populations over generations

plt.xlabel('Generation')
plt.ylabel('Average magnitude')
plt.plot(np.arange(500),a_totals, color='b')
plt.plot(np.arange(500),b_totals, color='r')
plt.show()
