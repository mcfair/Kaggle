import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy
"""
Step1: compute distance
Step2: assign the cluster label to each x,  
Step3: recompute the centroid for each cluster
Repeat step2 &3 
"""
def kmeans(X, k, distfunc='euclidean', tol=1e-3):
    """
    X: np 2D array
    k: integer
    distfunc: distance metric  'euclidean'
    """
    X = StandardScaler().fit_transform(X)
    y = np.zeros(len(X))
    prev_centers = np.random.rand(k,2)
   
    error = float('inf')
    
    nloops=0
    while error >= tol :
         
        dist = scipy.spatial.distance.cdist(X, prev_centers, metric=distfunc)
        y = np.argmin(dist, axis=1)
            
        centers = np.zeros((k,2))
        for i in range(k):
            centers[i]= np.mean(X[y==i], axis=0)
            
            
        error = np.mean(scipy.spatial.distance.cdist(centers, prev_centers, metric=distfunc))
        prev_centers = centers
        nloops+=1
         
    return y



if __name__ == '__main__':
    X = np.random.rand(150, 2)
    y = kmeans(X, 5, distfunc='euclidean', tol=1e-1)
    
    for i in range(len(y)):
        print (X[i], y[i])
