import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import numpy as np
import pdb
import pandas as pd
from pygam import ExpectileGAM
import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    white_viridis = LinearSegmentedColormap.from_list(
        'white_viridis',
        [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ],
        N=256,
        gamma=1,
    )
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')


def expreg(X, y, svrarray, index):
    # lets fit the mean model first by CV
    # X = X[~(np.isnan(X))]
    ind = y.isin([np.nan, np.inf, -np.inf]).any(1)
    X = X[~ind]
    y = y[~ind]
    svrarray = svrarray[~ind]
    X = np.sort(X)
    X.shape = (X.shape[0], 1)
    #    gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)
    # and copy the smoothing to the other models
    lam = 100
    # print(lam)
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X, y)
    gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
    gam90 = ExpectileGAM(expectile=0.9, lam=lam).fit(X, y)
    gam02 = ExpectileGAM(expectile=0.02, lam=lam).fit(X, y)
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X, y)
    gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)
    pred99 = gam99.predict(X)
    pred95 = gam95.predict(X)
    pred02 = gam02.predict(X)
    pred05 = gam05.predict(X)
    pred90 = gam90.predict(X)
    pred005 = gam005.predict(X)
    plt.ylim(0, 3000)
    fig = plt.figure()
    using_mpl_scatter_density(fig, X, svrarray)
    plt.plot(X, pred90, label='90')
    plt.plot(X, pred005, label='005')
    plt.plot(X, pred02, label='02')
    plt.plot(X, pred95, label='95')
    plt.plot(X, pred99, label='99')
    plt.plot(X, pred05, label='05')

    plt.legend()
    plt.savefig('%s.png' % index)
    plt.close()

    # i=i+1


pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir11 = '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band11 = [file for file in files if re.match(swir11, file)]
band12 = [file for file in files if re.match(swir12, file)]
len = 300
ii = 0


final_svr = np.array([])
final_ndvi = np.array([])

for file1, file2, file3, file4 in zip(band4, band8, band11, band12):
    if ii == 3:
        ii += 1
        continue
    n1 = range(len)
    img4 = mpimg.imread('./data/' + file1)
    img8 = mpimg.imread('./data/' + file2)
    img11 = mpimg.imread('./data/' + file3)
    img12 = mpimg.imread('./data/' + file4)
    img4 = img4[n1, :]
    img4 = img4[:, n1]
    img8 = img8[n1, :]
    img8 = img8[:, n1]
    img12 = img12[n1, :]
    img12 = img12[:, n1]
    img11 = img11[n1, :]
    img11 = img11[:, n1]
    img8.astype('float64')
    img4.astype('float64')
    img12.astype('float64')
    img11.astype('float64')
    #    x = img8 + img4
    ragh = []
    test1 = []
    test2 = []
    for i in range(0, len):
        t1 = np.copy(img8[i]).astype('float64')
        t2 = np.copy(img4[i]).astype('float64')
        den = t1 + t2
        temp = np.true_divide(abs(t1 - t2), den, out=np.zeros_like(den), where=den != 0)
        ragh.append(temp)
        t3 = np.copy(img11[i].astype('float64'))
        t4 = np.copy(img12[i].astype('float64'))

        den1 = 2 * t3
        den2 = 2 * t4

        temp1 = np.true_divide(
            ((1 - t3) ** 2), den1, out=np.zeros_like(den1), where=den1 != 0
        )

        temp2 = np.true_divide(
            ((1 - t4) ** 2), den2, out=np.zeros_like(den2), where=den2 != 0
        )

        test1.append(temp1)
        test2.append(temp2)
    pdb.set_trace()
    ndvi = np.array(ragh)
    svr1 = np.array(test1)
    svr2 = np.array(test2)
    # ndvi = np.divide(img8 - img4, x)
    invalid = (ndvi > 1).any()
    if invalid:
        pdb.set_trace()

    #    img11 = mpimg.imread('./data/'+file3)
    #   img12 = mpimg.imread('./data/'+file4)
    #  print(img8)
    #   print(img4)
    # svr1 = np.divide(((1 - img11)**2), (2 * img11))
    # svr2 = np.divide(((1 - img12)**2), (2 * img12))
    print('done')
    #  save_plot(svr1, svr2, ndvi, ii)
    # ii += 1
    # print(ii)
    ndvi = np.reshape(ndvi, -1)
    #    print(ndvi)
    svr1 = np.reshape(svr1, -1)
    svr2 = np.reshape(svr2, -1)
    final_svr = np.concatenate((final_svr, svr2))
    final_ndvi = np.concatenate((final_ndvi, ndvi))
    # df_ndvi = pd.DataFrame(ndvi, columns=['ndvi'])
    # df_svr1 = pd.DataFrame(svr1, columns=['svr1'])
    # df_svr2 = pd.DataFrame(svr2, columns=['svr1'])
    ii += 1
    print(ii)

ind1 = np.argwhere(np.isnan(final_svr) | np.isinf(final_svr))
ind2 = np.argwhere(np.isnan(final_ndvi) | np.isinf(final_ndvi))
ind = np.unique(np.concatenate((ind1, ind2)))
print("***********\n", ind.shape)

print("Writing data to csv")
rows = zip(final_ndvi, final_svr)

with open('data.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
# print(final_svr.shape)
# print(final_svr_reduced.shape)
# print(final_ndvi.shape)
# print(final_ndvi_reduced.shape)
# try:
#     expreg(ndvi, df_svr2, svr2, "ra" + file1[-12:-4])
# except ValueError as e:
#     print(e)
#     if str(e) != 'y data should have at least 1 samples, but found 0':
#         raise e
