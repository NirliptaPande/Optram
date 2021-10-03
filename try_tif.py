import numpy as np
from matplotlib import pyplot as plt
import rasterio
from multiprocessing import Pool
#plt.rcParams.update({'font.size': 160})

row = 10980
col = 10980

# len_row = (row // 300) + 1
# len_col = (col // 300) + 1


def compute(dummy,test):
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
    #fig = plt.figure(figsize=(109.8, 109.8), dpi=100)
    #plt.imshow(data)
    #plt.colorbar()
    #plt.savefig("soil%s.tiff' % test[-12:-4]")
    inds = np.where(data<0)
    data[inds]=0
    with rasterio.open('soil%s.tiff' % test[-12:-4], 'w', **profile) as f:
        f.write(data)
    del data
if __name__ == "__main__":
    swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
    files = os.listdir('data/')
    files = sorted(files)
    band12 = [file for file in files if re.match(swir12, file)]
    #temp_file = np.empty((10980, 10980))
    profile = {}
    a_args = range(0,35)
    with rasterio.open('./data/' + band12[0]) as f:
        profile = f.profile
    with Pool() as pool:
        pool.starmap(compute, zip(a_args, band12))

