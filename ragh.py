import os
import numpy as np
import re
import pdb
import matplotlib.image as mpimg
from pygam import ExpectileGAM


def expreg(X, y):
    try:
        X_arr = X.to_numpy()
    except:
        continue
    del X
    print('Converted to numpy')
    lam = 100
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    print('calc gam0.5')
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    print('predicting:')
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)
    return max(pred99), max(pred005)


pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band12 = [file for file in files if re.match(swir12, file)]
ii = 0

temp_file = mpimg.imread('./data/' + band4[0])
data = np.zeros((temp_file.shape[0], temp_file.shape[1]))


for len1 in range(0, temp_file.shape[0], 300):
    for len2 in range(0, temp_file.shape[1], 300):
        print(len1, '\t', len2, '\n')
        final_svr = np.array([])
        final_ndvi = np.array([])

        file1 = band4[0]
        file2 = band8[0]
        file3 = band12[0]
        img4 = mpimg.imread('./data/' + file1)
        img8 = mpimg.imread('./data/' + file2)
        img12 = mpimg.imread('./data/' + file3)
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

        img4 = img4[n1, :]
        img4 = img4[:, n2]
        img8 = img8[n1, :]
        img8 = img8[:, n2]
        img12 = img12[n1, :]
        img12 = img12[:, n2]
        img8.astype('float64')
        img4.astype('float64')
        img12.astype('float64')
        ragh = []
        test2 = []
        for i in range(0, 300):
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
        invalid = (ndvi > 1).any()
        if invalid:
            pdb.set_trace()
        final_ndvi = np.reshape(ndvi, -1)
        final_svr = np.reshape(svr2, -1)
        wet, dry = expreg(final_ndvi, final_svr)
        soil_data = (svr2 - dry) / (wet - dry)  # Shape = (300*300,1)
        soil_data = np.reshape(soil_data, (300, 300))
        data[len1 : len1 + 300, len2 : len2 + 300] = soil_data

        ii += 1
        print(ii)

print("Shape\t", data.shape)
mpimg.imsave('soil1.tiff', data)
