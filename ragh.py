import os
import numpy as np
import pandas as pd
import re
import pdb
import matplotlib.image as mpimg
from pygam import ExpectileGAM


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
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band12 = [file for file in files if re.match(swir12, file)]
ii = 0

temp_file = mpimg.imread('./data/' + band4[0])

# Create 35 empty variables
data = {}
for i in range(35):
    data[i] = np.zeros((temp_file.shape[0], temp_file.shape[1]))

for len1 in range(0, temp_file.shape[0], 300):
    for len2 in range(0, temp_file.shape[1], 300):
        if len1 + 300 > temp_file.shape[0]:
            test_len1 = temp_file.shape[0]
        else:
            test_len1 = len1 + 300
        if len2 + 300 > temp_file.shape[1]:
            test_len2 = temp_file.shape[1]
        else:
            test_len2 = len2 + 300

        final_svr = np.zeros(35 * (test_len1 - len1) * (test_len2 - len2))  # 1d array
        final_ndvi = np.zeros(35 * (test_len1 - len1) * (test_len2 - len2))

        n1 = range(len1, test_len1)
        n2 = range(len2, test_len2)

        for j in range(35):
            file1 = band4[j]
            file2 = band8[j]
            file3 = band12[j]
            print("Reading files for band %d" % j)
            img4 = mpimg.imread('./data/' + file1)
            img8 = mpimg.imread('./data/' + file2)
            img12 = mpimg.imread('./data/' + file3)

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

            print("Calculating NDVI")
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
            print("Finished ndvi, svr calculation")

            # Flatten the ndvi and svr
            temp_ndvi = np.reshape(temp_ndvi, -1)
            temp_svr = np.reshape(temp_svr, -1)
            temp_len = temp_ndvi.shape[0]
            final_ndvi[j * temp_len : (j + 1) * temp_len] = temp_ndvi
            final_svr[j * temp_len : (j + 1) * temp_len] = temp_svr

        if temp_file.shape[0] - len1 < 300:
            temp_row = temp_file.shape[0] - len1
        else:
            temp_row = 300

        if temp_file.shape[1] - len2 < 300:
            temp_col = temp_file.shape[1] - len2
        else:
            temp_col = 300

        if not final_svr.any():
            # Empty array
            soil_data = np.zeros(
                (temp_row, temp_col)
            )  # Should it be 0 or something else?
        else:
            df_svr = pd.DataFrame(final_svr, columns=['svr'])
            del final_svr
            print("Calc expecreg")
            wet, dry = expreg(final_ndvi, df_svr)
            print("Expecreg done")
            del final_ndvi
            if (wet - dry).all() == 0:
                soil_data = np.zeros((35, temp_row, temp_col))
            else:
                print("calc soil data")
                soil_data = (final_svr - dry) / (wet - dry)  # Shape = (300*300*35,1)
                print("done")
                np.reshape(soil_data, (35, temp_row, temp_col))

        for _ in range(35):
            data[_][len1 : len1 + 300, len2 : len2 + 300] = soil_data[_]

        ii += 1
        print("ii", "\n\n")


for i in range(35):
    print(data[i].shape)
    mpimg.imsave('clefinal_soil%i.tiff' % i, data[i])
