
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import numpy as np
def save_plot(svr1,svr2,ndvi,index):
    ndvi=np.reshape(ndvi,-1)
    print(ndvi)
    svr1 =np.reshape(svr1,-1)
    svr2=np.reshape(svr2,-1)
    plt.scatter(ndvi,svr1)
    plt.savefig('%stest.png'%index)
    plt.close()
    plt.scatter(ndvi,svr2)
    plt.savefig('%stest1.png'%index)
    plt.close()

    
pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir11= '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
swir12= '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band11 = [file for file in files if re.match(swir11, file)]
band12 = [file for file in files if re.match(swir12, file)]
i=0
for file1,file2,file3,file4 in zip(band4,band8,band11,band12):
    n1 = range(50)
    img4 = mpimg.imread('./data/'+file1)
    img8 = mpimg.imread('./data/'+file2)
    img11 = mpimg.imread('./data/'+file3)
    img12 = mpimg.imread('./data/'+file4)
    img4 = img4[n1,:]
    img4 = img4[:,n1]
    img8 = img8[n1,:]
    img8 = img8[:,n1]
    img12 = img12[n1,:]
    img12 = img12[:,n1]
    img11 = img11[n1,:]
    img11 = img11[:,n1]
    ndvi = (img8-img4)
    ndvi = np.divide(ndvi,(img8+img4))
#    img11 = mpimg.imread('./data/'+file3)
 #   img12 = mpimg.imread('./data/'+file4)
  #  print(img8)
 #   print(img4)
    svr1 = np.divide(((1-img11)**2),(2*img11))
    svr2= np.divide(((1-img12)**2),(2*img12))
    print('done')
    save_plot(svr1,svr2,ndvi,i)
    i+=1



