import csv
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
#swir11= '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
swir12= '^s2tile_31UDR_R051-N28_stack_s2-B12_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
#band11 = [file for file in files if re.match(swir11, file)]
band12 = [file for file in files if re.match(swir12, file)]
eps = 0.000001
del files
for file1,file2,file3 in zip(band4,band8,band12):
    img4 = mpimg.imread('./data/'+file1)
    img8 = mpimg.imread('./data/'+file2)
    ndvi = (img8-img4)/(img8+img4+eps)
#    img11 = mpimg.imread('./data/'+file3)
    img12 = mpimg.imread('./data/'+file3)
#    svr1 = (1-img11)**2/(2*img11+eps)
    svr2= (1-img12)**2/(2*img12+eps)
#    print(svr1.shape)
    print(ndvi.shape)
    rows = zip(svr1,ndvi)
    with open('file1.text','a') as f1:
        writer = csv.writer(f1)
        for row in rows:
            	writer.writerows(row)
    
    rows = zip(svr2,ndvi)
    with open('file2.txt','a') as f1:
        writer = csv.writer(f1)
        for row in rows:
                writer.writerows(row)

   # print('done')
   # plt.savefig('test.png')
    #plt.savefig('test1.png')
    print('done')
    del img4,img8,ndvi,img11,img12,svr1,svr2
