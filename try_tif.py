import numpy as np
from matplotlib import pyplot as plt
import rasterio

#plt.rcParams.update({'font.size': 160})

row = 10980
col = 10980

# len_row = (row // 300) + 1
# len_col = (col // 300) + 1


for dummy in range(35):
    data = np.zeros((row, col))

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

            wet_final = np.load('vars/wet_%d_%d.npy' % (len1, len2))
            wet_final = np.array_split(wet_final, 35)
            wet = np.reshape(wet_final[dummy], ((test_len1 - len1), (test_len2 - len2)))
            del wet_final
            dry_final = np.load('vars/dry_%d_%d.npy' % (len1, len2))
            dry_final = np.array_split(dry_final, 35)
            dry = np.reshape(dry_final[dummy], ((test_len1 - len1), (test_len2 - len2)))
            del dry_final
            svr_final = np.load('vars/svr_%d_%d.npy' % (len1, len2))
            svr_final = np.array_split(svr_final, 35)
            svr = np.reshape(svr_final[dummy], ((test_len1 - len1), (test_len2 - len2)))
            del svr_final

            soil_data = (svr - dry) / (wet - dry)

            data[len1:test_len1, len2:test_len2] = soil_data

    del soil_data, wet, dry, svr
    print(dummy)
    fig = plt.figure(figsize=(109.8, 109.8), dpi=100)
    plt.imshow(data)
    plt.colorbar()
    plt.savefig("soil%s.tiff' % file3[-12:-4]")
    del data
