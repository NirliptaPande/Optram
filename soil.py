from pygam import ExpectileGAM

def expreg(X, y, svrarray):
    X_arr = X.to_numpy()
    del X
    y_arr = y.to_numpy()
    print('Converted to numpy')
    lam = 100
    gam99 = ExpectileGAM(expectile=0.99, lam=lam).fit(X_arr, y)
    print('calc gam0.5')
    gam005 = ExpectileGAM(expectile=0.005, lam=lam).fit(X_arr, y)
    print('predicting:')
    pred99 = gam99.predict(X_arr)
    pred005 = gam005.predict(X_arr)
    