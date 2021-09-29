import os
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

files = os.listdir('data/')
# npyfiles = os.listdir('vars/')
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
band12 = [file for file in files if re.match(swir12, file)]
temp_file = mpimg.imread('./data/' + band12[0])
row = temp_file.shape[0]
col = temp_file.shape[1]
del temp_file

# len_row = (row // 300) + 1
# len_col = (col // 300) + 1

c = 0
for file3 in band12:
    data = np.zeros((row, col))
    print("Reading file: %s" % file3)
    img12_ = mpimg.imread('./data/' + file3)
    for len1 in range(0, row, 300):
        for len2 in range(0, col, 300):
            print(len1, '\t', len2)
            test_len1 = 0
            test_len2 = 0
            if len1 + 300 > row:
                test_len1 = row
            else:
                test_len1 = len1 + 300
            if len2 + 300 > col:
                test_len2 = col
            else:
                test_len2 = len2 + 300
            n1 = range(len1, test_len1)
            n2 = range(len2, test_len2)
            img12 = img12_[n1, :]
            img12 = img12[:, n2]
            svr = []
            print("Loading wet final")
            wet_final = np.load('vars/wet_%d_%d.npy' % (len1, len2))
            wet_final = np.array_split(wet_final, 35)
            wet = np.reshape(wet_final[c], ((test_len1 - len1), (test_len2 - len2)))
            del wet_final
            print("Loading dry final")
            dry_final = np.load('vars/dry_%d_%d.npy' % (len1, len2))
            dry_final = np.array_split(dry_final, 35)
            dry = np.reshape(dry_final[c], ((test_len1 - len1), (test_len2 - len2)))
            del dry_final

            print("computing svr")
            for i in range(0, img12.shape[0]):
                t4 = np.copy(img12[i].astype('float64'))
                den2 = 2 * t4
                temp2 = np.true_divide(
                    ((1 - t4) ** 2), den2, out=np.zeros_like(den2), where=den2 != 0
                )
                svr.append(temp2)
            soil_data = (svr - dry) / (wet - dry)
            data[len1:test_len1, len2:test_len2] = soil_data

    del soil_data, wet, dry, img12, svr, n1, n2
    print("\n")
    c = c + 1
    mpimg.imsave('clefinal_soil%s.tiff' % file3[-12:-4], data)
    np.save('vars/soil_%s' % file3[-12:-4], data)
    plt.imshow(data)
    plt.colorbar()
    plt.savefig('soil%s.tiff' % file3[-12:-4])
