import numpy as np

from random import seed
from random import sample
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn import neighbors

import math 

n_neighbors = 1



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

def distmatrices(X,XR):
    x2 =np.sum(X**2,axis = 1)
    y2 = np.sum(XR**2,axis=1)
    xy = np.matmul(X,XR.T)
    x2 = x2.reshape(-1,1)
    return np.sqrt(x2-2*xy+y2)
    
def vector2matrix(Xi,XR):
    return np.linalg.norm(Xi-XR,axis=1)
 

def distpond(Xi,Yi,XR,YR):
    n0=get_index_labels(Yi,0)
    n1=get_index_labels(Yi,1)
     
    n0= len(n0)
    n1=len(n1)
    n,m =Xi.shape
    for i in range(n+1):
        dE = vector2matrix(Xi[i],XR)
        if (Yi[i]==0):
            dE = ((n0/(n0+n1))**m)*dE
        else:
            dE = ((n1/(n0+n1))**m)*dE
        print(dE)



ftra = '03subcl5-600-5-70-bi-5-1tra.prn'
ftst = '03subcl5-600-5-70-bi-5-1tst.prn'

Xtra,ytra = load_dataset(ftra)
Xtest,ytest = load_dataset(ftst)

R,Ry = getR(Xtra,ytra,25,4)

newtra = distmatrices(Xtra,R)
newtst = distmatrices(Xtest,R)


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', algorithm="brute")

clf.fit(Xtra,ytra)

ypred = clf.predict(Xtest)

tpr,tnr=recall_score(ytest,ypred,average = None)
prec = precision_score(ytest,ypred, pos_label=0)
f1=f1_score(ytest,ypred, pos_label=0)
gmean = math.sqrt(tpr*tnr)
acc = accuracy_score(ytest,ypred)

print(tpr,tnr,prec,f1,gmean,acc)


newtra = distmatrices(Xtra,R)
newtst = distmatrices(Xtest,R)


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', algorithm="brute")

clf.fit(newtra,ytra)

ypred = clf.predict(newtst)

tpr,tnr=recall_score(ytest,ypred,average = None)
prec = precision_score(ytest,ypred, pos_label=0)
f1=f1_score(ytest,ypred, pos_label=0)
gmean = math.sqrt(tpr*tnr)
acc = accuracy_score(ytest,ypred)

print(tpr,tnr,prec,f1,gmean,acc)

distpond(Xtra,ytra,R,YR)


#idx0=get_index_labels(y,0)
#print(get_samples_class(X,idx0))
#idx1=get_index_labels(y,1)

#print(y[idx0])
#print(y[idx1])

#print(get_unique_labels(y))
