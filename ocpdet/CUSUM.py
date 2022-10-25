class CUSUM():
    """Cumulative Sum algorithm.
    
    Read more in 
    
    Parameters
    ----------
    
    k : float, default=0.25
        Control parameter of the CUSUM algorithm monitoring the gap between 
        the normalised stream and the algorithm statistics. 
    
    h : float, default=8.
        Control parameter of the CUSUM algorithm used as a threshold for the 
        decision rule.
    
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
    
    S : ndarray of shape (n_samples,)
        S statistic calculated sequentially after processing each new observation
        which measures increases.
    
    T : ndarray of shape (n_samples,)
        T statistics calculated sequentially after processing each new observation
        which measures decreases.
    
    changepoints : ndarray
        Array containing detected changepoints, initialised empty.
    
    n : int
        Number of observations in the current run length.
    """
    
    def __init__(self, 
                 k: float = 0.25, 
                 h: float = 8., 
                 burnin: int = 50, 
                 mu: float = 0., 
                 sigma:float = 1.):
        self.k = k
        self.h = h
        self.burnin = burnin
        self.mu = mu
        self.sigma = sigma

        self._mu = [mu]
        self._sigma = [sigma]
        self.S = [0.]
        self.T = [0.]
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
                          data_new: float):
        """Update the algorithm statistics S and T.

        Parameters
        ----------
        
        data_new : float
            New observation in the data stream. S and T are updated according to
            CUSUM algorithm formulas.
        """
        self.S.append(max(0, self.S[-1] + (data_new - self.mu) / self.sigma - self.k))
        self.T.append(max(0, self.S[-1] - (data_new - self.mu) / self.sigma - self.k))
    
    def decision_rule(self, 
                      i: int):
        """Decide whether or not a change has occurred.

        Parameters
        ----------
            
        i : int
            Time index. The decision rule S > h or T > h is implemented in this method.
        """
        if (i >= self.burnin) and ((self.S[-1] > self.h) or (self.T[-1] > self.h)):
            self.changepoints.append(i)
            self.S[-1] = 0
            self.T[-1] = 0
            self.n = 2
        else:
            self.n += 1
        
    def process(self, 
                data: list):
        """Run CUSUM algorithm on a univariate data stream.

        Parameters
        ----------
            
        data : list
            Univariate data stream to be processed. The method sequentially first updates mean 
            and variance, then updates the S and T statistics and finally applies the simple
            decision rule to assert if a change has occurred.
        """
        for i in range(1, len(data)):
            self.update_mean_variance(data[i])
            self.update_statistics(data[i])
            self.decision_rule(i)