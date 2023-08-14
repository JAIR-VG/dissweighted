import numpy as np

from random import seed
from random import sample




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

def get_samples_class(xsamples,idxsamples):
    return(xsamples[idxsamples,])

def dist2R(x_i, xr, yr, pi_pos, pi_neg):
    m,n = xr.shape
    for i in range(m):
        dxr=np.linalg.norm(xr-x_i,axis=1)
        if yr[i] == 0:
            dxr = pow(pi_pos,(1/n))*dxr
        else:
            dxr = pow(pi_neg,(1/n))*dxr


def getR(X,y,porcentaje,semilla):
    
    idx0=get_index_labels(y,0)
    idx1=get_index_labels(y,1)

    nsamp0 = round(len(idx0)*(porcentaje/100))
    nsamp1= round(len(idx1)*(porcentaje/100))
    seed(semilla)

    idxselected0 = sample(idx0,nsamp0)
    idxselected1 = sample(idx1,nsamp1)
    A =get_samples_class(X,idxselected0)
    B = get_samples_class(X,idxselected1)
    R = np.concatenate((A,B),axis=0)
    yra = y[idxselected0]
    yrb= y[idxselected1]
    YR = np.concatenate((yra,yrb),axis=0)
    return R, YR
    
 



ftra = '03subcl5-600-5-0-bi-5-1tra.prn'
ftst = '03subcl5-600-5-0-bi-5-1tst.prn'

X,y = load_dataset(ftra)

R,Ry = getR(X,y,10,4)
print(X[0])
print(len(R))

dx =np.linalg.norm(X[0]-R,axis=1)

print(dx)
print(len(dx))


#idx0=get_index_labels(y,0)
#print(get_samples_class(X,idx0))
#idx1=get_index_labels(y,1)

#print(y[idx0])
#print(y[idx1])

#print(get_unique_labels(y))
