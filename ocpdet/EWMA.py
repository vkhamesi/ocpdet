class EWMA():
    """Exponentially Weighted Moving Average algorithm.
        
    Read more in 
    
    Parameters
    ----------
    
    r : float, default=0.1
        Control parameter of the EWMA algorithm monitoring the learning rate
        of the exponentially moving weighted average Z (between 0 and 1).
    
    L : float, default=2.4.
        Control parameter of the EWMA algorithm used as a threshold for the 
        decision rule and controlling the bandwith.
    
    burnin : int, default=50
        Number of firts observed values processed before a changepoint can be
        detected. 
    
    mu : float, default=0.
        Initial mean value of the stream. Recall that CUSUM assumes that 
        observations are normally distributed.
    
    sigma : float, default=1.
        Initial standard deviation of the stream.
    
    Attributes
    ----------
    
    _mu : ndarray of shape (n_samples,)
        Recording of all mean values.
    
    _sigma : ndarray of shape (n_samples,)
        Recording of all standard deviation values.
    
    Z : ndarray of shape (n_samples,)
        Z statistic calculated sequentially after processing each new observation, 
        smoothing the original data stream.
    
    sigma_Z : ndarray of shape (n_samples,)
        Standard deviation of the Z statistic.
    
    changepoints : ndarray
        Array containing detected changepoints, initialised empty.
    
    n : int
        Number of observations in the current run length.
    """
    
    def __init__(self, 
                 r: float = 0.1, 
                 L: float = 2.4, 
                 burnin: int = 50, 
                 mu: float = 0., 
                 sigma:float = 1.):
        self.r = r
        self.L = L
        self.burnin = burnin
        self.mu = mu
        self.sigma = sigma

        self._mu = [mu]
        self._sigma = [sigma]
        self.Z = [mu]
        self.sigma_Z = [0.]
        self.changepoints = []
        self.n = 2
        
    def update_mean_variance(self, 
                             data_new: float):
        """Update efficiently mean and variance.

        Parameters
        ----------
        
        data_new : float
            New observation in the data stream. The mean and variance are updated
            efficiently and online without storing every observed values. 
        """
        mu_new = self.mu + (data_new - self.mu) / self.n
        self.sigma = (self.sigma ** 2 + ((data_new - self.mu) * (data_new - mu_new) - self.sigma ** 2) / self.n) ** 0.5
        self.mu = mu_new
        self._mu.append(mu_new)
        self._sigma.append(self.sigma)
    
    def update_statistics(self, 
                          i: int, 
                          data_new: float):
        """Update the algorithm statistics Z and sigma_Z.

        Parameters
        ----------
        
        i : int
            Time index.
        
        data_new : float
            New observation in the data stream. Z and sigma_Z are updated according to
            EWMA algorithm formulas.
        """
        self.Z.append((1 - self.r) * self.Z[-1] + self.r * data_new)
        self.sigma_Z.append(self.sigma * ((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * i))) ** 0.5)
    
    def decision_rule(self, 
                      i: int):
        """Decide whether or not a change has occurred.

        Parameters
        ----------
            
        i : int
            Time index. The decision rule |Z - mu| / sigma_Z > L is implemented in this method.
        """
        if (i >= self.burnin) and (abs((self.Z[-1] - self.mu) / self.L) > self.sigma_Z[-1]):
            self.changepoints.append(i)
            self.n = 2
        else:
            self.n += 1
        
    def process(self, 
                data: list):
        """Run EWMA algorithm on a univariate data stream.

        Parameters
        ----------
            
        data : list
            Univariate data stream to be processed. The method sequentially first updates mean 
            and variance, then updates the Z and sigma_Z statistics and finally applies the simple
            decision rule to assert if a change has occurred.
        """
        for i in range(1, len(data)):
            self.update_mean_variance(data[i])
            self.update_statistics(i, data[i])
            self.decision_rule(i)