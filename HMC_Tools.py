import time
import numpy as np
import matplotlib.pyplot as plt

class EvaluateHMC:
    def __init__(self, samples):
        self.chain = samples
        self.nsamples = samples.shape[0]
        
    def autocorrelation(self, lag):
        M = self.nsamples
        mean = np.mean(self.chain)
        var = np.sum((self.chain - mean) ** 2) / M
        return np.sum((self.chain[:M-lag] - mean) * (self.chain[lag:] - mean)) / (M - lag) / var

    def autocorrelation_function(self, max_lag):
        return np.array([self.autocorrelation(lag) for lag in range(max_lag)])

    def integrated_autocorrelation_time(self, autocorr, M):
        # Use a cutoff when autocorrelation lower than 0.05
        cuttoff_autocorr = autocorr[autocorr > 0.05]
        M_cutoff = len(cuttoff_autocorr)

        sum = 0
        for s in range(M_cutoff):
            sum += cuttoff_autocorr[s] * (1 - s/M)

        return 1 + 2 * sum

    def effective_sample_size(self, max_lag):
        M = self.nsamples
        autocorr = self.autocorrelation_function(max_lag)
        tau = self.integrated_autocorrelation_time(autocorr, M)
        ess = M / tau
        return ess 
    

    def plot_autocorrelation(self, max_lag, label, save_file=None):
        fig = plt.figure(figsize=(10, 10))
        autocorr = self.autocorrelation_function(max_lag)
        lags = np.arange(max_lag)

        plt.figure(figsize=(10, 5))
        plt.stem(lags, autocorr, linefmt='lightblue')
        plt.xlabel('Lag')
        plt.ylabel(f'{label} Autocorrelation')
        plt.title('Autocorrelation Function')
        plt.axhline(y=0.05, color='g', linestyle='--', linewidth=1)
        plt.axhline(y=-0.05, color='g', linestyle='--', linewidth=1)
        plt.grid(True)
        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()