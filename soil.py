from pygam import ExpectileGAM

def expreg(X, y):
    X_arr = X.to_numpy()
    del X
    print('Converted to numpy')
    lam = 100
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    print('calc gam0.5')
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    print('predicting:')
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)
    return max(pred99), max(pred005)
length = (len(band4)//300)+1
wet_final = [length][length]
dry_final = [length][length]
for len1 in in range(0, len(band4), 300):
    for len2 in range(0, len(band4), 300):
        data = pd.read_csv("data%%.csv"%(len1,len2))
        ndvi = data['0.0']
        svr = data['0.0.1']
        del data
        print('calling expecreg')
        wet_final[len1//300][len2//300], dry_final[len1//300][len2//300] = expreg(ndvi, svr)
for i in range(35):
    #soil%i[len(band4)][len(band4)]
    #basically create 35 arrays
for len1 in in range(0, len(band4), 300):
    for len2 in range(0, len(band4), 300):
        data = pd.read_csv("data%%.csv"%(len1,len2))
        wet = wet_final[len1//300][len2//300]
        dry = dry_final[len1//300][len2//300]
        data = (data-dry)/(wet-dry)
        #split it into 35 parts 
        #reshape each into a 300 by 300
        #place each colum in the appropriate place in the larger array
#convert it all into raster files