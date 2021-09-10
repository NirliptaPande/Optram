import re
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
#swir11= '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
swir12= '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
eps = 0.000001

band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
#band11 = [file for file in files if re.match(swir11, file)]
band12 = [file for file in files if re.match(swir12, file)]
file1 = band4[0]
file2 = band8[0]
file3 = band12[0]
i=1
for file1,file2,file3 in zip(band4,band8,band12):
    img4 = mpimg.imread('./data/'+file1)
    img8 = mpimg.imread('./data/'+file2)
    img12 = mpimg.imread('./data/'+file3)
    #     img8 = mpimg.imread('./data/'+file2)
    n1 = range(200)
    img4 = img4[n1,:]
    img4 = img4[:,n1]
    img8 = img8[n1,:]
    img8 = img8[:,n1]
    img12 = img12[n1,:]
    img12 = img12[:,n1]
    ndvi = (img8-img4)/(img8+img4+eps)
    str2= (1-img12)**2/(2*img12+eps)
   # ndvi = ndvi.flatten()
    #str2 = str2.flatten()
    plt.scatter(ndvi,str2,c='green',marker='x',alpha=0.02)
    plt.title("NDVI- STR scatter plot",fontsize=20)
    plt.xlabel("NDVI",fontsize=15)
    plt.ylabel("STR",fontsize=15)
    plt.savefig(str(i)+"NDVI-STR.jpg")
    i=i+1
