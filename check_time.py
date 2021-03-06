import os
import numpy as np
import re
import timeit
import matplotlib.image as mpimg


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
        tic = timeit.default_timer()
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

        for j in range(5, 20):
            file1 = band4[j]
            file2 = band8[j]
            file3 = band12[j]
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
        toc = timeit.default_timer()
        print('\n\n', toc - tic, '\n\n')
