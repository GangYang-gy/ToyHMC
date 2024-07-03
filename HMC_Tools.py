{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class EvaluateHMC:\n",
    "    def __init__(self, samples):\n",
    "        self.chain = samples\n",
    "        self.nsamples = samples.shape[0]\n",
    "        \n",
    "    def autocorrelation(self, lag):\n",
    "        M = self.nsamples\n",
    "        mean = np.mean(self.chain)\n",
    "        var = np.sum((self.chain - mean) ** 2) / M\n",
    "        return np.sum((self.chain[:M-lag] - mean) * (self.chain[lag:] - mean)) / (M - lag) / var\n",
    "\n",
    "    def autocorrelation_function(self, max_lag):\n",
    "        return np.array([self.autocorrelation(lag) for lag in range(max_lag)])\n",
    "\n",
    "    def integrated_autocorrelation_time(self, autocorr, M):\n",
    "        # Use a cutoff when autocorrelation lower than 0.05\n",
    "        cuttoff_autocorr = autocorr[autocorr > 0.05]\n",
    "        M_cutoff = len(cuttoff_autocorr)\n",
    "\n",
    "        sum = 0\n",
    "        for s in range(M_cutoff):\n",
    "            sum += cuttoff_autocorr[s] * (1 - s/M)\n",
    "\n",
    "        return 1 + 2 * sum\n",
    "\n",
    "    def effective_sample_size(self, max_lag):\n",
    "        M = self.nsamples\n",
    "        autocorr = self.autocorrelation_function(max_lag)\n",
    "        tau = self.integrated_autocorrelation_time(autocorr, M)\n",
    "        ess = M / tau\n",
    "        return ess \n",
    "    \n",
    "\n",
    "    def plot_autocorrelation(self, max_lag, label, save_file=None):\n",
    "        fig = plt.figure(figsize=(10, 10))\n",
    "        autocorr = self.autocorrelation_function(max_lag)\n",
    "        lags = np.arange(max_lag)\n",
    "\n",
    "        plt.figure(figsize=(10, 5))\n",
    "        plt.stem(lags, autocorr, linefmt='lightblue')\n",
    "        plt.xlabel('Lag')\n",
    "        plt.ylabel(f'{label} Autocorrelation')\n",
    "        plt.title('Autocorrelation Function')\n",
    "        plt.axhline(y=0.05, color='g', linestyle='--', linewidth=1)\n",
    "        plt.axhline(y=-0.05, color='g', linestyle='--', linewidth=1)\n",
    "        plt.grid(True)\n",
    "        if save_file:\n",
    "            plt.savefig(save_file)\n",
    "        else:\n",
    "            plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "work",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
