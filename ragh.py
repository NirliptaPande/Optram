import matplotlib.pyplot as plt
import matplotlib.image as mpimg
≈import re
import os
import numpy as np
import sklearn.linear_model
import pdb
import pandas as pd

def save_plot(svr1, svr2, ndvi, index):
#    ndvi = np.reshape(ndvi, -1)
#    print(ndvi)
  #  svr1 = np.reshape(svr1, -1)
 #  svr2 = np.reshape(svr2, -1)
    plt.scatter(ndvi, svr1, c= 'green', alpha = 0.05)
    plt.savefig('%stest.png' % index)
    plt.close()
    plt.scatter(ndvi, svr2,c = 'purple', alpha = 0.05)
    plt.savefig('%stest1.png' % index)
    plt.close()
def quantile_reg(df_ndvi, df_svr1, df_svr2):
    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
 #out_bounds_predictions = np.zeros_like(y_true_mean, dtype=np.bool_)
    for quantile in quantiles:
    	qr = QuantileRegressor(quantile=quantile, alpha=0)
    	y_pred = qr.fit(df_ndvi,df_svr2).predict(df_ndvi)
    	predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
    	plt.plot(X, y_pred, label=f"Quantile: {quantile}")
    plt.scatter(
    df_ndvi,
    df_svr1,
    color="black",
    marker="+",
    alpha=0.05,
)
    plt.savefig('%squantile.png' %index) 
pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir11 = '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
swir12 = '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band11 = [file for file in files if re.match(swir11, file)]
band12 = [file for file in files if re.match(swir12, file)]
len = 500
ii = 0
for file1, file2, file3, file4 in zip(band4, band8, band11, band12):
    n1 = range(len)
    img4 = mpimg.imread('./data/' + file1)
    img8 = mpimg.imread('./data/' + file2)
    img11 = mpimg.imread('./data/' + file3)
    img12 = mpimg.imread('./data/' + file4)
#    print(img4.shape)
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
        temp = np.true_divide(abs(t1 - t2), (t1 + t2 + 0.00001))
        ragh.append(temp)
        t3 = np.copy(img11[i].astype('float64'))
        t4 = np.copy(img12[i].astype('float64'))
        temp1 = np.true_divide(((1-t3)**2),(2*t3))
        temp2 = np.true_divide(((1-t4)**2),(2*t4))
        test1.append(temp1)
        test2.append(temp2)
    ndvi = np.array(ragh)
    svr1 = np.array(test1)
    svr2 = np.array(test2)
    #ndvi = np.divide(img8 - img4, x)

    invalid = (ndvi > 1).any()
    if invalid:
        pdb.set_trace()

#    img11 = mpimg.imread('./data/'+file3)
 #   img12 = mpimg.imread('./data/'+file4)
  #  print(img8)
 #   print(img4)
    #svr1 = np.divide(((1 - img11)**2), (2 * img11))
    #svr2 = np.divide(((1 - img12)**2), (2 * img12))
    print('done')
    save_plot(svr1, svr2, ndvi, ii)
    ii += 1
    print(ii)
    ndvi = np.reshape(ndvi, -1)
#    print(ndvi)
    svr1 = np.reshape(svr1, -1)
    svr2 = np.reshape(svr2, -1)
    df_ndvi = pd.DataFrame(ndvi, columns = ['ndvi'])
    df_svr1 = pd.DataFrame(svr1)
    df_svr2 = pd.DataFrame(svr2)
    quantile_reg(df_ndvi, df_svr1, df_svr2)
