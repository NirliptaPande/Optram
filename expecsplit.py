import os
import numpy as np
from pandas import DataFrame
import re
import pdb
from pygam import ExpectileGAM
import rasterio

def expreg(X, y):
    try:
        X_arr = X.to_numpy()
    except:
        X_arr = X
    del X
    lam = 100
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)
    return pred99, pred005


pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
files = sorted(files)
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band12 = [file for file in files if re.match(swir12, file)]

ii = 0
temp_file = np.empty((10980, 10980))
with rasterio.open('./data/' + band4[0]) as f:
    temp_file = f.read(1)
row = temp_file.shape[0]
col = temp_file.shape[1]
del temp_file
img4_ = {}
img8_ = {}
img12_ = {}
#profile = {}

for len1 in range(0, row, 300):
    for len2 in range(0, col, 300):
        if len1 + 300 > row:
            test_len1 = row
        else:
            test_len1 = len1 + 300
        if len2 + 300 > col:
            test_len2 = col
        else:
            test_len2 = len2 + 300

        final_svr = np.zeros(35 * (test_len1 - len1) * (test_len2 - len2),dtype=float)  # 1d array
        final_ndvi = np.zeros(35 * (test_len1 - len1) * (test_len2 - len2),dtype=float)

        n1 = range(len1, test_len1)
        n2 = range(len2, test_len2)

        for j in range(35):
            file1 = band4[j]
            file2 = band8[j]
            file3 = band12[j]
            with rasterio.open('./data/' + file1) as f:
                img4 = mpimg.imread('./data/' + file1)
            with rasterio.open('./data/' + file2) as f:
                img8 = mpimg.imread('./data/' + file2)
            with rasterio.open('./data/' + file1) as f:
                img12 = mpimg.imread('./data/' + file3)
                print(f.profile)
                print(j)
            img4 = img4[n1, :]
            img4 = img4[:, n2]
            img8 = img8[n1, :]
            img8 = img8[:, n2]
            img12 = img12[n1, :]
            img12 = img12[:, n2]

            img8.astype('float64')
            img4.astype('float64')
            img12.astype('float64')
            temp_ndvi = np.zeros_like(img4)
            temp_svr = np.zeros_like(img4)

            # Calculate NDVI row-wise
            for i in range(0, img4.shape[0]):
                t1 = np.copy(img8[i]).astype('float64')
                t2 = np.copy(img4[i]).astype('float64')
                den = t1 + t2
                temp = np.true_divide(
                    abs(t1 - t2), den, out=np.zeros_like(den), where=den != 0
                )  # NDVI of a row
                temp_ndvi[i] = temp
                t4 = np.copy(img12[i].astype('float64'))
                den2 = 2 * t4
                temp2 = np.true_divide(
                    ((1 - t4) ** 2), den2, out=np.zeros_like(den2), where=den2 != 0
                )  # SVR of a row
                temp_svr[i] = temp2
            temp_ndvi = np.array(temp_ndvi)
            temp_svr = np.array(temp_svr)

            # Flatten the ndvi and svr
            temp_ndvi = np.reshape(temp_ndvi, -1)
            temp_svr = np.reshape(temp_svr, -1)
            temp_len = temp_ndvi.shape[0]
            final_ndvi[j * temp_len : (j + 1) * temp_len] = temp_ndvi
            final_svr[j * temp_len : (j + 1) * temp_len] = temp_svr
            
        df_svr = DataFrame(final_svr, columns=['svr'])#use directly pd.dataframe in import
        wet, dry = expreg(final_ndvi, df_svr)
        #feather.write_feather(df_svr, 'vars/svr_%d_%d.feather' % (len1, len2))
        #feather.write_feather(wet, 'vars/wet_%d_%d.feather' % (len1, len2))
        #feather.write_feather(dry, 'vars/dry_%d_%d.feather' % (len1, len2))
        np.save('vars/svr_%d_%d.npy' % (len1, len2),final_svr)
        np.save('vars/wet_%d_%d.npy' % (len1, len2),wet)
        np.save('vars/dry_%d_%d.npy' % (len1, len2),dry)
