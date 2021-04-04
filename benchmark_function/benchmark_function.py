"""
@author: Yuki Shimizu
Benchmarck functions for optimization
"""

import math

import numpy as np


# check dimension
def check(x):
    x = np.array(x)
    if len(x.shape)!=2: raise Exception('Only 2D array is expected. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.')
    return x

# benchmark function
class BenchmarkFunction:
    """
    Parameter------
    input_array : 
        2D array of shape (n, m) is expected.
        n : number of data
        m : input size
    ---------------
    
    Functions------
    sphere : 
        Basic function
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)
    
    ellipsoid : 
        Weak ill-scale
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)
    
    k_tablet : 
        strong ill-scale
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)

    rosenbrock_star :
        Strong parameter dependency between x1 and the others
        Search area: [-2.048,2.048]^n
        Optimal solusion: (1,...,1)

    rosenbrock_chain : 
        Strong parameter dependency between neighboring parameters
        Search area: [-2.048,2.048]^n
        Optimal solusion: (1,...,1)
    
    bohachevsky : 
        Weak multimodality
        Search area: [-5.12,5.12]^n
        Optimal solusion: (0,...,0)

    ackley : 
        Weak multimodality
        Search area: [-32.768,32.768]^n
        Optimal solusion: (0,...,0)

    schaffer : 
        Strong multimodality
        Search area: [-100,100]^n
        Optimal solusion: (0,...,0)

    rastrigin : 
        Strong multimodality
        Search area: [-5.12,5.12]^n
        Optimal solusion: (1,...,1)
    ---------------
    
    search_area------
    Return the search areas (type: dict) for each function 
    ---------------

    optimal_solution------
    Return the optimal solutions (type: dict) for each function 
    dimension (int) :
        dimension of input variable
    ---------------
    """
    def __init__(self):
        pass
    
    def sphere(self, input_array): 
        input_array = check(input_array)
        return (input_array**2).sum(axis=-1)
    
    def ellipsoid(self, input_array): 
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        coef = [1000**(i/(dim-1)) for i in range(dim)]
        coef = np.array(coef)
        
        return ((input_array*coef)**2).sum(axis=1)
    
    def k_tablet(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        k = math.ceil(dim/4)
                
        return (input_array[:,:k]**2).sum(axis=1) \
                +((100*input_array[:,k:])**2).sum(axis=1)
        
    def rosenbrock_star(self, input_array):
        input_array = check(input_array)
        input_array_1st = input_array[:,:1]
        input_array_rest = input_array[:,1:]
        
        return (100*(input_array_1st-input_array_rest**2)**2 \
                +(1-input_array_rest)**2).sum(axis=1)

    def rosenbrock_chain(self, input_array):
        input_array = check(input_array)
        
        return (100*(input_array[:,1:]-input_array[:,:-1]**2)**2 \
                +(1-input_array[:,:-1])**2).sum(axis=1)
        
    def bohachevsky(self, input_array):
        input_array = check(input_array)
        
        return (input_array[:,:-1]**2 \
                +2*input_array[:,1:]**2 \
                -0.3*np.cos(3*np.pi*input_array[:,:-1]) \
                -0.4*np.cos(4*np.pi*input_array[:,1:]) \
                +0.7).sum(axis=1)
        
    def ackley(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        return 20 \
                -20*np.exp(-0.2*((input_array**2).sum(axis=1)/dim)**0.5) \
                +np.e \
                -np.exp((np.cos(2*np.pi*input_array)).sum(axis=1)/dim)

    def schaffer(self, input_array):
        input_array = check(input_array)
        
        return ((input_array[:,:-1]**2+input_array[:,1:]**2)**0.25 \
                 *(np.sin(50*(input_array[:,:-1]**2+input_array[:,1:]**2)**0.1)**2 \
                +1.0)).sum(axis=1)

    def rastrigin(self, input_array):
        input_array = check(input_array)
        dim = input_array.shape[1]
        
        return 10*dim + \
                ((input_array-1)**2-10*np.cos(2*np.pi*(input_array-1))).sum(axis=1)

    def search_area(self):
        search_area = {
            'sphere' : [-5.12,5.12],
            'ellipsoid' : [-5.12,5.12],
            'k_tablet' : [-5.12,5.12],
            'rosenbrock_star' : [-2.048,2.048],
            'rosenbrock_chain' : [-2.048,2.048],
            'bohachevsky' : [-5.12,5.12],
            'ackley' : [-32.768,32.768],
            'schaffer' : [-100,100],
            'rastrigin' : [-5.12,5.12],
        }
        return search_area
    
    def optimal_solution(self, dimension=2):
        zeros = [0]*dimension
        ones = [1]*dimension
        
        optimal_solution = {
            'sphere' : zeros,
            'ellipsoid' : zeros,
            'k_tablet' : zeros,
            'rosenbrock_star' : ones,
            'rosenbrock_chain' : ones,
            'bohachevsky' : zeros,
            'ackley' : zeros,
            'schaffer' : zeros,
            'rastrigin' : ones,
        }
        return optimal_solution
    
