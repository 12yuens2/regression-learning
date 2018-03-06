import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(df, feature_list):
    path = "histograms/"

    for feature in feature_list:
        ax = df.hist(feature)
        fig = ax[0][0].get_figure()
        fig.savefig(path + feature + ".png")

    
