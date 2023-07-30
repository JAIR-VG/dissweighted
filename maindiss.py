import numpy as np



def load_dataset(nfile):
    df = np.loadtxt(nfile,dtype='str')
    return np.asarray(df[:,0:-1],dtype=float), np.asarray(df[:,-1],dtype=int)


def get_all_labels(ylabel,indx):
    return(ylabel[indx])

def get_unique_labels(ylabel):
    return np.unique(ylabel)

def get_index_labels(ylab, labelclass):
    get_indexes = lambda ylab, xs: [i for (y, i) in zip(xs, range(len(xs))) if ylab == y]
    return get_indexes(labelclass,ylab)


def dist2R(x_i, xr, yr, pi_pos, pi_neg):
   
    m,n = xr.shape
    for i in range(m):
        dxr=np.linalg.norm(xr-x_i,axis=1)
        if yr[i] == 0:
            dxr = pow(pi_pos,(1/n))*dxr
        else:
            dxr = pow(pi_neg,(1/n))*dxr


ftra = '03subcl5-600-5-0-bi-5-1tra.prn'
ftst = '03subcl5-600-5-0-bi-5-1tst.prn'

X,y = load_dataset(ftra)
print(get_index_labels(y,0))

print(get_unique_labels(y))
