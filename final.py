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
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band12 = [file for file in files if re.match(swir12, file)]
ii = 0

temp_file = mpimg.imread('./data/' + band4[3])
data = np.zeros((temp_file.shape[0], temp_file.shape[1]))


for j in range(35):
    if j < 10:
        j += 1
        continue
    file1 = band4[j]
    file2 = band8[j]
    file3 = band12[j]
    img4_ = mpimg.imread('./data/' + file1)
    img8_ = mpimg.imread('./data/' + file2)
    img12_ = mpimg.imread('./data/' + file3)

    for len1 in range(0, temp_file.shape[0], 300):
        for len2 in range(0, temp_file.shape[1], 300):
            final_svr = np.array([])
            final_ndvi = np.array([])

            if len1 + 300 > temp_file.shape[0]:
                test_len1 = temp_file.shape[0]
            else:
                test_len1 = len1 + 300
            if len2 + 300 > temp_file.shape[1]:
                test_len2 = temp_file.shape[1]
            else:
                test_len2 = len2 + 300
            n1 = range(len1, test_len1)
            n2 = range(len2, test_len2)

            img4 = img4_[n1, :]
            img4 = img4[:, n2]
            img8 = img8_[n1, :]
            img8 = img8[:, n2]
            img12 = img12_[n1, :]
            img12 = img12[:, n2]
            img8.astype('float64')
            img4.astype('float64')
            img12.astype('float64')
            ragh = []
            test2 = []
            for i in range(0, img4.shape[0]):
                t1 = np.copy(img8[i]).astype('float64')
                t2 = np.copy(img4[i]).astype('float64')
                den = t1 + t2
                temp = np.true_divide(
                    abs(t1 - t2), den, out=np.zeros_like(den), where=den != 0
                )
                ragh.append(temp)
                t4 = np.copy(img12[i].astype('float64'))
                den2 = 2 * t4
                temp2 = np.true_divide(
                    ((1 - t4) ** 2), den2, out=np.zeros_like(den2), where=den2 != 0
                )
                test2.append(temp2)
            ndvi = np.array(ragh)
            svr2 = np.array(test2)
            if not svr2.any():
                # Empty array
                soil_data = np.zeros_like(svr2)
            else:
                final_ndvi = np.reshape(ndvi, -1)
                final_svr = np.reshape(svr2, -1)
                df_svr = pd.DataFrame(final_svr, columns=['svr'])
                del final_svr
                wet, dry = expreg(final_ndvi, df_svr)
                wet = np.reshape(wet, svr2.shape)
                dry = np.reshape(dry, svr2.shape)
                if (wet - dry).all() == 0:
                    soil_data = np.zeros_like(svr2)
                else:
                    soil_data = (svr2 - dry) / (wet - dry)  # Shape = (300*300,1)

            if temp_file.shape[0] - len1 < 300:
                temp_row = temp_file.shape[0] - len1
            else:
                temp_row = 300

            if temp_file.shape[1] - len2 < 300:
                temp_col = temp_file.shape[1] - len2
            else:
                temp_col = 300

            try:
                soil_data = np.reshape(soil_data, (temp_row, temp_col))
            except:
                pdb.set_trace()

            data[len1 : len1 + 300, len2 : len2 + 300] = soil_data

            ii += 1

    print("****\n\nFILE %d \n\n*****" % (j + 1))
    mpimg.imsave('clefinal_soil%i.tiff' % (j + 1), data)
