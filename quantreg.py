import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pdb
import pandas as pd


def quantile_reg(df_ndvi, df_svr1):
    data = pd.concat([df_ndvi, df_svr1], axis =1)
#    print(data.head())
    data.columns = ['x', 'y']
    indices = data.isin([np.nan, np.inf, -np.inf]).any(1)
    data = data[~indices]
    df_svr1 = df_svr1[~indices]
    df_ndvi = df_ndvi[~indices]
    mod = smf.quantreg('y~x', data)
    mod1 = smf.quantreg('x~y', data)
    quantiles = np.array([0.05, 0.1, 0.9, 0.95])
    models = []
    models1 = []
    params = []
    params1 = []
    for qt in quantiles:
        res1 = mod1.fit(q = qt)
        res = mod.fit(q = qt)
        models.append(res)
        models1.append(res1)
        params1.append([qt, res1.params['Intercept'], res1.params['y']])
        params.append([qt, res.params['Intercept'], res.params['x']])

    params = pd.DataFrame(data = params, columns = ['qt', 'intercept', 'x_coef'])
    params1 = pd.DataFrame(data = params1, columns = ['qt', 'intercept', 'x_coef'])

#    print(models[0].summary())

    #res = model1.fit(q=.95)
    plt.scatter(df_ndvi, df_svr1, color='black', marker='x', alpha=0.02)
    y_pred1 = models[0].params['Intercept'] + models[0].params['x'] * df_ndvi
    plt.plot(df_ndvi, y_pred1, linewidth=2, label='Q Reg : 0.05')
    y_pred2 = models[1].params['Intercept'] + models[1].params['x'] * df_ndvi
    plt.plot(df_ndvi, y_pred2, linewidth=2, label='Q Reg : 0.1')
    y_pred3 = models[2].params['Intercept'] + models[2].params['x'] * df_ndvi
    plt.plot(df_ndvi, y_pred3, linewidth=2, label='Q Reg : 0.9')
    y_pred4 = models[3].params['Intercept'] + models[3].params['x'] * df_ndvi
    plt.plot(df_ndvi, y_pred4, color='yellow', linewidth=2, label='Q Reg : 0.95')
    pdb.set_trace()
    x_pred1 = models1[0].params['Intercept'] + models1[0].params['y'] * df_svr1
    plt.plot(x_pred1, df_svr1, linewidth=2, label='Q Reg : 0.05')
    x_pred2 = models1[1].params['Intercept'] + models1[1].params['y'] * df_svr1
    plt.plot(x_pred2, df_svr1, linewidth=2, label='Q Reg : 0.05')
    x_pred3 = models1[2].params['Intercept'] + models1[2].params['y'] * df_svr1
    plt.plot(x_pred3, df_svr1, linewidth=2, label='Q Reg : 0.05')
    x_pred4 = models1[3].params['Intercept'] + models1[3].params['y'] * df_svr1
    plt.plot(x_pred4, df_svr1, linewidth=2, label='Q Reg : 0.05')

    plt.title("NDVI- STR scatter plot", fontsize=20)
    plt.xlabel("NDVI", fontsize=15)
    plt.ylabel("STR", fontsize=15)
    plt.savefig("%sNDVI-STR.jpg" % ii)
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
len = 500
ii = 0

for file1, file2, file3, file4 in zip(band4, band8, band11, band12):
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
        temp = np.true_divide(abs(t1 - t2), (t1 + t2 + 0.00001))
        ragh.append(temp)
        t3 = np.copy(img11[i].astype('float64'))
        t4 = np.copy(img12[i].astype('float64'))
        temp1 = np.true_divide(((1 - t3)**2), (2 * t3))
        temp2 = np.true_divide(((1 - t4)**2), (2 * t4))
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
  #  save_plot(svr1, svr2, ndvi, ii)
   # ii += 1
    # print(ii)
    ndvi = np.reshape(ndvi, -1)
#    print(ndvi)
    svr1 = np.reshape(svr1, -1)
    svr2 = np.reshape(svr2, -1)
    df_ndvi = pd.DataFrame(ndvi, columns = ['ndvi'])
    df_svr1 = pd.DataFrame(svr1, columns = ['svr1'])
    df_svr2 = pd.DataFrame(svr2, columns = ['svr1'])
    ii += 1
    print(ii)
    try:
        quantile_reg(df_ndvi, df_svr2)
    except ValueError as e:
        print("******")
        print(e)
        continue
   # ii += 1
  #  print(ii)
