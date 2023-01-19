import numpy as np
from scipy.stats import ranksums, mood, mannwhitneyu, ks_2samp, cramervonmises_2samp

class TwoSample():
    """Two sample test for changepoint detection.
    
    Read more in
    
    Parameters
    ----------
    
    statistic : str, default="Lepage"
        Test statistic to be used by the algorithm. Use 'Mann-Whitney' for changes
        in the location, 'Mood' for changes in the scale, 'Lepage' for changes in 
        both location and scale, 'Kolmogorov-Smirnov' and 'Cramer-von-Mises' for 
        general changes in distribution.
    
    threshold : float, default=3.1
        Threshold value for the test statistic, must be suited for each statistic.
        
    Attributes
    ----------
    
    t : int
        Time index.
    
    changepoints : list
        Array containing detected changepoints, initialised empty. 
    """
    
    def __init__(self, 
                 statistic: str = "Lepage", 
                 threshold: float = 3.1):
        self.threshold = threshold
        self.statistic = statistic
        
        self.D = [1.]
        self.t = 2
        self.changepoints = []
        
    def fetch_statistic(self):
        # Maps statistic attribute with SciPy method
        db = {
            "Mann-Whitney": mannwhitneyu, 
            "Mood": mood, 
            "Lepage": ranksums, 
            "Kolmogorov-Smirnov": ks_2samp,
            "Cramer-von-Mises": cramervonmises_2samp
        }
        self.statistic = db[self.statistic]
    
    def process_batch(self, 
                      X: list):
        """Process a data stream until a change is detected.

        Parameters
        ----------
            
        X : list
            Univariate array to be processed.

        Returns
        -------
        
        tau : int or None
            First detected changepoint in X. If no change is detected, the method
            returns None.
        """
        Dn = []
        tau = 0
        self.t = 2
        while self.t < len(X):
            Dkn = []
            for k in range(1, self.t):
                x, y = X[tau:k], X[k:self.t]
                # Different syntax for each Scipy test
                try:
                    Dkn.append(abs(self.statistic(x, y)[0]))
                except:
                    try:
                        Dkn.append(abs(self.statistic(x, y).statistic))
                    except: # Edge cases with sample sizes 
                        Dkn.append(0)
            self.D.append(max(Dkn))
            if max(Dkn) < self.threshold:
                Dn.append(max(Dkn))
                self.t += 1
            else:
                tau = self.t
                return tau
        return None

    def process(self, 
                data: list):
        """Run the two-sample test algorithm.

        Parameters
        ----------
            
        data : list
            Univariate data stream to be processed. The method first tries to detect
            a change in the data. If a change is detected, it will look for changes in the
            remaining sequence, etc. until no change is detected. When no change is detected
            and the whole sequence has been processed, the method stops.
        """
        self.fetch_statistic()
        cp = []
        tau = 0
        while tau is not None:
            data = data[tau:]
            tau = self.process_batch(data)
            self.D.append(self.D[-1])
            cp.append(tau)
        self.changepoints = np.cumsum(cp[:-1])