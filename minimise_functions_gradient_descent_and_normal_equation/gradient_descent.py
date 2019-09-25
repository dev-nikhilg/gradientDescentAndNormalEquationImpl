import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    
    def __init__(self, builder):
        """Gradient descent class to minimise a generic function 
        
        Attributes:
            builder : an instance of the builder class
            
        """
        
        if builder.X:
            self.X = builder.X
            self.num_examples = self.X.shape[0]
            self.num_features = self.X.shape[1] - 1
        else:
            self.num_examples = 100
            self.num_features = 1
            self.X = self.default_single_feature_X()
            
            
        if builder.Y:
            self.Y = builder.Y
        else:
            self.Y = self.default_linear_related_Y()
     
        
        self.theta_vector = None
        
        
    def default_single_feature_X(self):
        """ Method to create some some sample values of X
            we add an additional first column of 1s as needed in the gradient descent
            
        Args: 
            None
        
        Returns:
            return a 100*2 matrix with first column made of all 1s 
            and second column represents random observed values of the only feature involved
            
        """
        
        X = 2 * np.random.rand(self.num_examples, self.num_features)
        ones = np.ones((self.num_examples, 1))
        return np.concatenate((ones, X), axis=1)
    
    
    
    def default_linear_related_Y(self):
        """ Method to generate some linearly related values to the observed feature matrix X
        
        Args:
            None
            
        Returns:
            Returns a vector with randomised values linearly related to X. 
            Number of vales/rows in vector Y is equal to the number of observations (number of rows) in feature matrix X
            
        """
        
        Y = np.random.randn(self.num_examples, 1)

        for i in range(self.num_features + 1):
            rand = np.random.randint(1,10)
            col = self.X[:,i].reshape(self.num_examples, 1)
            Y = np.add(Y, col * rand)
        
        #Plot the curve
        plt.plot(self.X[:, 1], Y, 'ro')
        plt.show()
        
        return Y
    
    
    def default_theta_vector(self):
        """ Method the generate initial values of theta parameter vector
        
        Args:
            None
            
        Return:
            Return a vector with all values initialised to 0
            
        """
        
        return np.zeros((self.num_features + 1, 1))
    
    
    def minimise(self, theta_vector = None, alpha = 0.5, threshold = 0.05):
        """ Method which starts the gradient descent algorithm
        
        Args:
            theta_vector : Initial value of theta_vector to use in the algorithm
            
        Return:
            theta_vector : theta_value vector corresponding to our best hypothesis
            
        """
        
        if theta_vector:
            self.theta_vector = theta_vector
        else:
            self.theta_vector = self.default_theta_vector()
            
        num_iterations = 0
        prev_cost = float(0)
        
        hypothesis_output_vector = self.calculate_hypothesis_output()
        hypothesis_cost = self.calculate_hypothesis_cost(hypothesis_output_vector)
        
        while abs(hypothesis_cost - prev_cost) > threshold:
            num_iterations = num_iterations + 1
            
            self.theta_vector = self.calculate_new_theta_vector(hypothesis_output_vector, alpha)
            
            hypothesis_output_vector = self.calculate_hypothesis_output()
            prev_cost = hypothesis_cost
            hypothesis_cost = self.calculate_hypothesis_cost(hypothesis_output_vector)

            
        #Plot the curve
        plt.plot(self.X[:, 1], self.Y, 'ro')
        plt.plot(self.X[:, 1], hypothesis_output_vector)
        plt.show()
            
            
            
    def calculate_hypothesis_output(self):
        """ Method to calculate the output vector of our current hypothesis 
            which is represented by the theta_vector
            
        Args:
            None
            
        Returns:
            h(theta_vector) : vector of predicted values for observed feature matrix X
            
        """
        
        return np.matmul(self.X, self.theta_vector)
                
        
    def calculate_hypothesis_cost(self, hypothesis_output_vector):
        """ Method to calculate the output vector of our current hypothesis 
            which is represented by the theta_vector
            
        Args:
            hypothesis_output_vector : vector containing predictions for our input feature matrix X and current theta_vector
            
        Returns:
            float : cost of current hypothesis as compared to Y
            
        """
        
        cost = float(0)

        for index in range(self.num_examples):
            cost = cost + ((hypothesis_output_vector[index][0] - self.Y[index][0]) ** 2)
        
        return cost
        
        
    def calculate_new_theta_vector(self, hypothesis_output_vector, alpha):
        """ Method to calculate new values for the theta_vector based on current hypothesis
        
        Args:
            hypothesis_output_vector : current hypothesis output vector
            alpha : learning rate
            
        Returns:
            theta_vector : vector containing new values of theta
            
        """
        
        new_theta_vector = self.theta_vector
        
        for index in range(self.num_features + 1):
            diff_term = hypothesis_output_vector - self.Y
            diff_term = np.multiply(diff_term, self.X[:,index].reshape(self.num_examples, 1))
            derivative_term = 1.0 * alpha * np.sum(diff_term) / self.num_examples
            
            new_theta_vector[index][0] = new_theta_vector[index][0] - derivative_term
        
        
        return new_theta_vector
        
        
        
    class Builder:
        
        def __init__(self):
            """ Builder class for the GradientDescent class. Initialises all variables with null
            
            Attributes: None
        
            """
            self.X = None
            self.Y = None
            self.theta_vector = None 
        
        
        def setX(self, X):
            """ Builder method to set the value of feature matrix X
            
            Args:
                X : feature matrix X
                
            Returns:
                self : self Builder instance
                
            """
            
            self.X = X
            return self
        
        
        def setY(self, Y):
            """ Builder method to set the value of observed output values vector Y
            
            Args:
                Y : observed output values vector Y
                
            Returns:
                self : self Builder instance
                
            """
            
            self.Y = Y
            return self
        
        
        def build(self):
            """ Builder method used to create the instance of GradientDescent
            
            Args:
                None
                
            Returns:
                GradientDescent : a new instance of GradientDescent
                
            """
            
            return GradientDescent(self)
                     
      
builder = GradientDescent.Builder()

a = builder.build()
a.minimise()