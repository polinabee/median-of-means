import numpy as np
import pandas as pd
import random
import statistics
import seaborn as sns
import matplotlib.pyplot as plt


class meanEstimator():
    def __init__(self, sample, l, N, epsilon):
        self.sample = sample  # sample from distribution
        self.n = len(sample)  # population size
        self.l = l  # number of chunks
        self.N = N  # number of shuffles
        self.epsilon = epsilon  # accuracy param
        self.empmean = sum(self.sample) / self.n
        self.probs = []

    def shuffle(self, seed):
        random_state = seed
        shuffled = self.sample.copy()
        random.shuffle(shuffled)
        return shuffled

    def median_of_means(self, shuffled=None):
        shuffled = self.sample if shuffled is None else shuffled
        chunks = np.array_split(np.array(shuffled), self.l)
        mom = statistics.median([statistics.mean(c) for c in chunks])
        p = 0
        if abs(mom - 0 ) <= self.epsilon:
            p = 1
        self.probs.append(p)
        return mom

    def perm_invar_mom(self):
        seeds = np.random.randint(40, size=self.N)
        shuffled_lists = [self.shuffle(s) for s in seeds]
        mom_arr = [self.median_of_means(l) for l in shuffled_lists]
        return statistics.mean(mom_arr)

    def get_p_mom(self):
        return statistics.mean(self.probs)

def various_n():
    reps = []
    g_mom_vals = []
    s_mom_vals = []
    g_mean_vals = []
    s_mean_vals = []
    g_mom_pi_vals = []
    s_mom_pi_vals = []

    for n in range(100,2000,20):
        reps.append(n)
        gdist = np.random.normal(0, 0.1, n)
        sdist = np.random.standard_t(10, size=n)
        gauss_estimator = meanEstimator(gdist, 10, 10, 0.01)
        student_estimator = meanEstimator(sdist, 10, 10, 0.01)

        g_mom_vals.append(gauss_estimator.median_of_means())
        g_mean_vals.append(gauss_estimator.empmean)
        g_mom_pi_vals.append(gauss_estimator.perm_invar_mom())

        s_mom_vals.append(student_estimator.median_of_means())
        s_mean_vals.append(student_estimator.empmean)
        s_mom_pi_vals.append(student_estimator.perm_invar_mom())

    return reps,g_mom_vals,s_mom_vals,g_mean_vals,s_mean_vals,g_mom_pi_vals,s_mom_pi_vals

def various_N():
    reps = []
    g_mom_vals = []
    s_mom_vals = []
    g_mean_vals = []
    s_mean_vals = []
    g_mom_pi_vals = []
    s_mom_pi_vals = []

    for N in range(5, 500, 10):
        reps.append(N)
        gdist = np.random.normal(0, 0.1, 1000)
        sdist = np.random.standard_t(10, size=1000)
        gauss_estimator = meanEstimator(gdist, 10, N, 0.01)
        student_estimator = meanEstimator(sdist, 10, N, 0.01)

        g_mom_vals.append(gauss_estimator.median_of_means())
        g_mean_vals.append(gauss_estimator.empmean)
        g_mom_pi_vals.append(gauss_estimator.perm_invar_mom())

        s_mom_vals.append(student_estimator.median_of_means())
        s_mean_vals.append(student_estimator.empmean)
        s_mom_pi_vals.append(student_estimator.perm_invar_mom())

    return reps, g_mom_vals, s_mom_vals, g_mean_vals, s_mean_vals, g_mom_pi_vals, s_mom_pi_vals


def vals_to_df(reps,g_mom_vals,s_mom_vals,g_mean_vals,s_mean_vals,g_mom_pi_vals,s_mom_pi_vals):
    frame = {'reps': pd.Series(reps),
             'gauss_mom': pd.Series(g_mom_vals),
             'student_mom': pd.Series(s_mom_vals),
             'gauss_mean': pd.Series(g_mean_vals),
             'student_mean': pd.Series(s_mean_vals),
             'gauss_pi_mom': pd.Series(g_mom_pi_vals),
             'student_pi_mom': pd.Series(s_mom_pi_vals)}
    return pd.DataFrame(frame)

if __name__ == '__main__':
    mu, sigma = 0, 0.1  # mean and standard deviation
    gauss_iid = np.random.normal(mu, sigma, 1000)

    estimator = meanEstimator(gauss_iid, 10, 10, 0.01)
    mom = estimator.median_of_means()
    perm_invar_mom = estimator.perm_invar_mom()

    a,b,c,d,e,f,g = various_n()
    sns.lineplot(x='reps', y='value', hue='variable',data=pd.melt(vals_to_df(a,b,c,d,e,f,g), ['reps']))
    plt.title('Means over sample size, increments of 20')
    plt.show()

    a, b, c, d, e, f, g = various_N()
    sns.lineplot(x='reps', y='value', hue='variable', data=pd.melt(vals_to_df(a, b, c, d, e, f, g), ['reps']))
    plt.title('Means over number of shuffles, increments of 10')
    plt.show()


