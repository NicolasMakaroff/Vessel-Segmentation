import numpy as np

def fast_marching_2D(W, x0):
    n = W.shape[0]
    neigh = [[1,0],[-1,0],[0,1],[0,-1]]
    
    boundary = lambda x: x*(x<=n and x>0) + (2-x) * (x<=0) + (2*n-x)*(x>n)
    
    ind2sub1 = lambda k: [(k-1)%n +1, (k- (k-1)%n -1)/n +1]
    sub2ind1 = lambda u: (u[2]-1)*n + u[1]
    
    Neigh = lambda k,i: sub2ind1(boundary(ind2sub1(k)+neigh[:,i]))
    
    I = sub2ind1(x0)
    
    D = np.zeros(n) + np.inf
    D[I] = 0
    
    S = np.zeros(n)
    s[I]=1
    
    while I is not []:
        [tmp,j] = np.sort(D[I])
        j = j[1]
        i = I[j]
        I[j] = []
        S[i] = -1
        
        J = [Neigh[i,1],Neigh[i,2],Neigh[i,3], Neigh[i,4]]
        
        J[S[J]==-1] = []
        
        J1 = J[S[J]==0]
        I = [I,J1]
        S[J1]=1
        
        for j in J:
            dx = np.min(D[[Neigh(j,1),Neigh[j,2]]])
            dy = np.min(D[[Neigh(j,3), Neigh[j,4]]])
            Delta = 2*W[j] - (dx-dy)**2
            if Delta >=0:
                D[j] = (dx+dy+np.sqrt(Delta))/2
            else:
                D[j] = np.min(dx+W[j], dy+W[j])
                              
    S = np.zeros(n,n)

    return D, S