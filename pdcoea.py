''' pairwise dominance co-evolutionary algorithm'''

import numpy as np


def create_population(pop_size,chromosome_len,real_valued=False,mean=None,stdv=None):
    '''
    creates a population of agents to initialize the pdcoea algorithm.

    Arguments:
        pop_size: The size of the population to be created.
        chromosome_len: Length of the chromosomes to be created.
        real_valued: If true, indicates that the chromosome is real valued, otherwise it is assumed to be boolean.
            If the chromosome is to be real-valued, the chromosomes are initialised using a normal distribution.
        mean: Only use if chromosomes are real valued. The mean of the normal dist used to initialize the chromosomes.
        stdv: Only use if chromosomes are real valued. The std dev of the normal dist used to initialize the chromosomes.


    Returns: list containing the created population of chromosomes.
    '''
    population = []
    for i in range(pop_size):
        if real_valued:
             chromosome = np.random.normal(mean,stdv,chromosome_len)
        else:
            chromosome = np.random.choice([0,1],chromosome_len)
            population.append(chromosome)
    return population


def dominance_test(x1,x2,y1,y2,g, **kwargs):
    '''
    given a two pairs of chromosomes, returns the dominant pair. 

    Arguments:
        x1, x2: two chromosomes from population 'a'
        y1, y2: two chromosomes from population 'b'
        g: the dominance function.

    Returns: the dominant chromosome pair.
    '''
    if (g(x1,y2,**kwargs) >= g(x1,y1,**kwargs)) and (g(x1,y1,**kwargs) >= g(x2,y1,**kwargs)):
        return [x1,y1]
    return [x2,y2]

def mutate(chromosome,chi,chromosome_len,real_valued=False):
    '''
    mutates a chromosome.

    Arguments:
        chromosome: the chromosome to mutate.
        chi: the probability of mutation parameter.
        chromosome_len: Length of the chromosomes.
        real_valued: If true, indicates that the chromosome is real valued, otherwise it is assumed to be boolean.

    Returns: the mutated chromosome.
    '''
    new_chromosome = []
    if real_valued:
        for gene in chromosome:
            if np.random.random() < (chi/chromosome_len):
                new_gene = gene + np.random.normal(0,0.1)
            else:
                new_gene = gene
            new_chromosome.append((new_gene))
    else:
        for gene in chromosome:
            if np.random.random() < (chi/chromosome_len):
                new_gene = abs(gene-1)
            else:
                new_gene = gene
            new_chromosome.append(int(new_gene))
    return new_chromosome

def pdcoea(pop_size,chromosome_len,epochs,g,sample=False,real_valued=False,mean=None,stdv=None, **kwargs):
    '''
    implements PDCo-EA algorithm, proposed by Lehre et. al. in https://link.springer.com/article/10.1007/s00453-024-01218-3.

    Arguments:
        pop_size: The size of the population.
        chromosome_len: Length of the chromosomes.
        epochs: number of generations to loop over.
        g: the dominance function to use to test for dominance.
        sample: Boolean if true, stores a sample from each generation to a list
        real_valued: If true, indicates that the chromosome is real valued, otherwise it is assumed to be boolean.
            If the chromosome is to be real-valued, the chromosomes are initialised using a normal distribution.
        mean: Only use if chromosomes are real valued. The mean of the normal dist used to initialize the chromosomes.
        stdv: Only use if chromosomes are real valued. The std dev of the normal dist used to initialize the chromosomes.
        
    Returns: the final populations.
    '''
    samples = []

    pop_a = create_population(pop_size,chromosome_len,real_valued,mean,stdv)
    pop_b = create_population(pop_size,chromosome_len,real_valued,mean,stdv)
    for i in range(epochs):
        new_a = []
        new_b = []
        for j in range(pop_size):
            x1 = pop_a[np.random.randint(len(pop_a))] 
            y1 = pop_b[np.random.randint(len(pop_b))] 
            x2 = pop_a[np.random.randint(len(pop_a))] 
            y2 = pop_b[np.random.randint(len(pop_b))]
            dom_pair = dominance_test(x1,x2,y1,y2,g,**kwargs)
            pair = [mutate(dom_pair[0],1,len(dom_pair[0])),mutate(dom_pair[1],1,len(dom_pair[1]))]
            new_a.append(pair[0])
            new_b.append(pair[1])
        pop_a = new_a
        pop_b = new_b
        if sample:
            samples.append((np.sum(pop_a)/chromosome_len,np.sum(pop_b)/chromosome_len))
    if samples:
        return pop_a, pop_b, samples
    return pop_a, pop_b






