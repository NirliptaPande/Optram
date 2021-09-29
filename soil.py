from pygam import ExpectileGAM
import pandas as pd
import os
import re
import matplotlib.image as mpimg
import numpy as np


def expreg(X, y):
    X_arr = X.to_numpy()
    del X
    #print('Converted to numpy')
    lam = 100
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    #print('calc gam0.5')
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    #print('predicting:')
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)
    return pred99, pred005 #change this


files = os.listdir('data/')
pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
band4 = [file for file in files if re.match(pattern1, file)]
temp_file = mpimg.imread('./data/' + band4[0])
row = temp_file.shape[0]
col = temp_file.shape[1]

len_row = (row // 300) + 1
len_col = (col // 300) + 1

wet_final = [len_row][len_col]
dry_final = [len_row][len_col]
for len1 in range(0, row, 300):
    for len2 in range(0, col, 300):
        data = pd.read_csv("data%%.csv" % (len1, len2))
        ndvi = data['0.0']
        svr = data['0.0.1']
        del data
        #print('calling expecreg')
        (
            wet_final[len1 // 300][len2 // 300],
            dry_final[len1 // 300][len2 // 300],
        ) = expreg(ndvi, svr)

data = {}
for i in range(35):
    data[i] = np.zeros((row, col))

for len1 in range(0, row, 300):
    for len2 in range(0, col, 300):
        data = pd.read_csv("data%%.csv" % (len1, len2))
        svr = data['0.0.1']
        wet = wet_final[len1 // 300][len2 // 300]
        dry = dry_final[len1 // 300][len2 // 300]
        soil_data = (svr - dry) / (wet - dry)
        # split it into 35 parts
        soil_data = np.reshape(soil_data, (35, 300, 300))
        for _ in range(35):
            data[_][len1 : len1 + 300, len2 : len2 + 300] = soil_data[_]

# Saving as rastor files
for i in range(35):
    print(data[i].shape)
    mpimg.imsave('soil%i.tiff' % i, data[i])
