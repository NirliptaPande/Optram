#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import ExpectileGAM
#import mpl_scatter_density  # adds projection='scatter_density'
#from matplotlib.colors import LinearSegmentedColormap

def expreg(X, y, index):
    X_arr = X.to_numpy()
    del X
    y_arr = y.to_numpy()
    print('Converted to numpy')
    lam = 100
    # print(lam)
   # print('calc gam99')
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    print('calc gam0.5')

    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    print('predicting:')
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)


data = pd.read_csv("data.csv")
print('read csv')

ndvi = data['0.0']
svr = data['0.0.1']
del data
print('calling expecreg')
expreg(ndvi, svr, 'rafinal')
