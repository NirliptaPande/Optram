import csv
import os
import numpy as np
import re
import pdb

pattern1 = '^s2tile_31UDR_R051-N28_stack_s2-B04_2018.....tif$'
pattern2 = '^s2tile_31UDR_R051-N28_stack_s2-B08_2018.....tif$'
swir11 = '^s2tile_31UDR_R051-N28_stack_s2-B11_2018.....tif$'
files = os.listdir('data/')
band4 = [file for file in files if re.match(pattern1, file)]
band8 = [file for file in files if re.match(pattern2, file)]
band12 = [file for file in files if re.match(swir12, file)]
ii = 0
final_svr = np.array([])
final_ndvi = np.array([])
for len1 in range(0,len(band4),300):
	for len2 in range(0,len(band4),300):
		for file1, file2, file3 in zip(band4, band8, band12):
			img4 = mpimg.imread('./data/' + file1)
		    img8 = mpimg.imread('./data/' + file2)
		    img12 = mpimg.imread('./data/' + file3)
		    n1 = range(len1, len1+300) 
		    n2 = range(len2, len2+300)#this line needs to be changed to move the window
		    img4 = img4[n1, :]
		    img4 = img4[:, n1]
		    img8 = img8[n1, :]
		    img8 = img8[:, n1]
		    img12 = img12[n1, :]
		    img12 = img12[:, n1]
		    img8.astype('float64')
		    img4.astype('float64')
		    img12.astype('float64')
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

		        den1 = 2 * t3

		        temp1 = np.true_divide(
		            ((1 - t3) ** 2), den1, out=np.zeros_like(den1), where=den1 != 0
		        )
		  

		        test1.append(temp1)
		    ndvi = np.array(ragh)
		    svr1 = np.array(test1)
		    invalid = (ndvi > 1).any()
		    if invalid:
		        pdb.set_trace()
		    ndvi = np.reshape(ndvi, -1)
		    #    print(ndvi)
		    svr1 = np.reshape(svr1, -1)
		    final_svr = np.concatenate((final_svr, svr1))
		    final_ndvi = np.concatenate((final_ndvi, ndvi))
		    ii += 1
		    print(ii)

		print("Writing data to csv")
		rows = zip(final_ndvi, final_svr)

		with open('data%s%s.csv'%(len1,len2), "w") as f:
		    writer = csv.writer(f)
		    for row in rows:
		        writer.writerow(row)


