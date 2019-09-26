import numpy as np
import matplotlib.pyplot as plt

class NormalEquation:
    """ NormalEquation class to minimise a generic function
        
    Attributes:
        X (n*m feature matrix) representing the feature matrix of the observations. 
            n is the number of observations
            m is the number of features in an observation 
        Y (n*1 vector) representing the observed output value of our observations
        
    """
    
    def __init__(self, X = None, Y = None):
        if X:
            self.X