import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotNormal

class NeuralNetwork():
    """Online changepoint detection using neural networks.
    
    Read more in
    
    Parameters
    ----------
    
    k : int, default=10
        First windowing parameter which stacks d-dimensional observations in arrays
        of shape (k, d) called autoregressive forms.
    
    n : int, default=5
        Second windowing parameter which stacks autoregressive observations of dimension
        (k, d) into tensors of shape (n, k, d). Those tensors are called mini-batches.
    
    lag : int, default=100
        Lag parameter controlling the time delay between a recent and a lagged mini-batch of
        data to be compared by the neural network. Recall that one should use n << lag.
    
    f : tf.keras.models.Sequential, default=None
        Neural network architecture to learn the mini-batches distribution sequentially. Must
        be a Keras object, with input size (n, k, d) and output size (1,). The default neural
        network architecture is a 1D Convolutional neural network with 2 layers, along with 
        BatchNormalisation layers. 
    
    r : float, default=0.1
        Learning rate of the modified EWMA algorithm. 
    
    L : float, default=3.
        Bandwith parameter of the modified EWMA algorithm.
    
    burnin : int, default=100
        Burnin period during which the first observations can not contain any changepoint.
    
    method : str, default='bump'
        The modified EWMA algorithm to use to detect changepoints in the dissimilarity sequence.
        Use 'bump' to detect sequential increases and decreases, use 'increase' to detect each bump
        increases.
    
    timeout : int, default=100
        Timeout period to be used if method='increase'. When a changepoint is detected, no change
        can be detected in the following timeout data observations.
    
    Attributes
    ----------
    
    divergence : ndarray
        Kullback-Leibler divergence sequence computed at each iteration and mini-batches comparison.
    
    dissimilarity : ndarray
        Rolling Kullback-Leibler divergence on the last samples to create a dissimilarity sequence 
        containing bumps.
        
    optimiser : tf.keras.optimizers, default=tf.keras.optimizers.Adam()
        Optimiser object used for the neural network training.
        
    Z : ndarray
        Modified EWMA algorithm exponentially weighted moving average statistic.
    
    sigma_Z : ndarray
        Modified EWMA algorithm standard deviation of the Z statistic.
    
    changepoints : ndarray
        Array containing detected changepoints, initialised empty.
    """
    
    def __init__(self, 
                 k: int = 10, 
                 n: int = 5, 
                 lag:  int = 100, 
                 f: tf.keras.models.Sequential = None, 
                 r: float = 0.1, 
                 L: float = 3., 
                 burnin: int = 100, 
                 method: str = "bump", 
                 timeout: int = 100):
        self.k = k
        self.n = n
        self.l = lag
        if f is None:
            # Neural network architecture by default
            self.f = Sequential([
                Conv1D(16, 2, activation="elu", input_shape=(self.n, self.k), kernel_initializer=GlorotNormal()),
                BatchNormalization(),
                Conv1D(8, 2, activation="elu", kernel_initializer=GlorotNormal()),
                BatchNormalization(),
                Flatten(),
                Dense(1, activation="sigmoid", kernel_initializer=GlorotNormal())
            ])
        else:
            self.f = f
        self.r = r
        self.L = L
        self.burnin = burnin
        self.timeout = timeout
        if method == "bump" or method == "increase":
            self.method = method
        else:
            print("Method can either be 'bump' or 'increase'.")
            self.method = "bump"
        
        self.divergence = []
        self.dissimilarity = []
        self.optimiser = Adam()
        self.Z = [0.]
        self.sigma_Z = [0.]
        self.changepoints = []
    
    def window_data(self, 
                    data: list):
        """Window the d-dimensional stream.

        Parameters
        ----------
        
        data : list
            d-dimensional data stream to be processed to create the mini-batches of data.
        """
        T = len(data)
        self.X = []
        for i in range(T - self.k + 1):
            self.X.append(data[i:self.k+i])
        self.X = np.asarray(self.X)
        self.Xi = []
        for i in range(T - self.k - self.n + 2):
            self.Xi.append(self.X[i:self.n+i])
        self.Xi = np.asarray(self.Xi)

    def compute_dissimilarity(self):
        """Compute the dissimilarity sequence.
        """
        for i in tqdm(range(len(self.Xi) - self.l - 1)):
            X_lagged = self.Xi[i][tf.newaxis, ...]
            X_recent = self.Xi[i+self.l][tf.newaxis, ...]
            d = tf.math.log((1 - self.f(X_lagged)) / self.f(X_lagged)) + tf.math.log(self.f(X_recent) / (1 - self.f(X_recent)))
            if i <= self.l + 1:
                self.dissimilarity.append(0.)
            else:
                d_bar = self.dissimilarity[-1] + (d - self.divergence[i-1-self.l]) / self.l
                self.dissimilarity.append(d_bar.numpy()[0][0])
            self.divergence.append(d.numpy()[0][0])
            with tf.GradientTape() as tape:
                loss_value = - tf.math.log(1 - self.f(X_lagged)) - tf.math.log(self.f(X_recent))
            grads = tape.gradient(loss_value, self.f.trainable_weights)
            self.optimiser.apply_gradients(zip(grads, self.f.trainable_weights))
        self.divergence = np.asarray(self.divergence)
        self.dissimilarity = np.asarray(self.dissimilarity)
        
    def detect_increase(self):
        """Detect increases in the dissimilarity.
        """
        n = 2
        mu_hat = 0.
        sigma_hat = 1.
        last_cp = 0
        for j in range(1, self.burnin):
            self.Z.append((1 - self.r) * self.Z[j-1] + self.r * self.dissimilarity[j])
            mu_hat_new = 1/n * ((n-1) * mu_hat + self.dissimilarity[j])
            sigma_hat = np.sqrt(1 / (n - 1) * ((self.dissimilarity[j] - mu_hat_new) * 
                                             (self.dissimilarity[j] - mu_hat) + (n - 2) * sigma_hat))
            mu_hat = mu_hat_new
            n += 1
            self.sigma_Z.append(sigma_hat * np.sqrt((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * j))))
        for j in range(self.burnin, len(self.dissimilarity)):
            self.Z.append((1 - self.r) * self.Z[j-1] + self.r * self.dissimilarity[j])
            mu_hat_new = 1/n * ((n-1) * mu_hat + self.dissimilarity[j])
            sigma_hat = np.sqrt(1 / (n - 1) * ((self.dissimilarity[j] - mu_hat_new) * 
                                             (self.dissimilarity[j] - mu_hat) + (n - 2) * sigma_hat))
            mu_hat = mu_hat_new
            n += 1
            self.sigma_Z.append(sigma_hat * np.sqrt((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * j))))
            if self.Z[j] > mu_hat + self.sigma_Z[j] * self.L:
                if j - last_cp > self.timeout:
                    self.changepoints.append(j)
                    last_cp = j
                n = 2
    
    def detect_bump(self):
        """Detect sequential increase and decrease in the dissimilarity.
        """
        n = 2
        mu_hat = 0.
        sigma_hat = 1.
        last = None
        last_det = None
        for j in range(1, self.burnin):
            self.Z.append((1 - self.r) * self.Z[j-1] + self.r * self.dissimilarity[j])
            mu_hat_new = 1/n * ((n-1) * mu_hat + self.dissimilarity[j])
            sigma_hat = np.sqrt(1 / (n - 1) * ((self.dissimilarity[j] - mu_hat_new) * 
                                             (self.dissimilarity[j] - mu_hat) + (n - 2) * sigma_hat))
            mu_hat = mu_hat_new
            n += 1
            self.sigma_Z.append(sigma_hat * np.sqrt((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * j))))
        for j in range(self.burnin, len(self.dissimilarity)):
            self.Z.append((1 - self.r) * self.Z[j-1] + self.r * self.dissimilarity[j])
            mu_hat_new = 1/n * ((n-1) * mu_hat + self.dissimilarity[j])
            sigma_hat = np.sqrt(1 / (n - 1) * ((self.dissimilarity[j] - mu_hat_new) * 
                                             (self.dissimilarity[j] - mu_hat) + (n - 2) * sigma_hat))
            mu_hat = mu_hat_new
            n += 1
            self.sigma_Z.append(sigma_hat * np.sqrt((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * j))))
            if self.Z[j] > mu_hat + self.sigma_Z[j] * self.L and last_det is None:
                last = "+"
                last_det = j
                n = 2
            if self.Z[j] < mu_hat - self.sigma_Z[j] * self.L and last == "+":
                last = "-"
                n = 2
                self.changepoints.append(last_det)
                last_det = None
        
    def process(self, 
                data: list):
        """Run the online changepoint detection using neural network algorithm.

        Parameters
        ----------
        
        data : list
            Multivariate or univariate data stream to be processed. The method sequentially
            computes the Kullback-Leibler divergence between a recent mini-batch of data and 
            a lagged mini-batch of data using the neural network, which learns the distribution.
            Changepoints are detected online by detecting bumps or increases in the outputted 
            dissimilarity measure. 
        """
        # Create mini-batches of data
        self.window_data(data)
        # Compute dissimilarity measure online from mini-batches
        self.compute_dissimilarity()
        # Detect changepoints in the dissimilarity sequence
        if self.method == "bump":
            self.detect_bump()
        elif self.method == "increase":
            self.detect_increase()
        # Translate time (detection times)
        self.changepoints = np.asarray(self.changepoints) + len(data) - len(self.dissimilarity)