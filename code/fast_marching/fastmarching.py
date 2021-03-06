import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import imageplot


def plot(img,propa,iter):
    plt.clf
    plt.imshow(img,cmap='gray')
    plt.imshow(propa)

    plt.savefig('fig_%i.png'%iter)


def perform_dijstra_fm(W, pstart, niter=np.inf, method='dijstr', bound='sym', svg_rate=1000):
    """
    % perform_fm_dijstra - *slow* (matlab) implementation of Dijstra and FM
    %
    %   [D,Dsvg,Ssvg] = perform_fm_dijstra(W, pstart, options);
    %
    %   W is an (n,n) metric matrix.
    %   pstart is a (2,k) starting points.
    %   options.method is either 'fm' or 'dijstra'
    %
    %   D is the final distance map to pstart
    %   options.svg_rate gives the rate at wich Dsvg and Ssvg is filled.
    %   options.niter can be used to limit the total number of steps (partial propagation).
    %
    %   Copyright (c) 2014 Gabriel Peyre
    """
    ##
    # Size.

    n = W.shape[0]

    ##
    # The four displacement vector to go to the four neightbors.

    neigh = np.array([[1, -1, 0, 0], [0, 0,  1, -1]])

    ##
    # For simplicity of implementation, we use periodic boundary conditions.

    # boundary = @(x)x.*(x<=n & x>0) + (2-x).*(x<=0) + (2*n-x).*(x>n);
    def symmetrize(x,n):
        if (x<0):
            x = -x;
        elif (x>=n):
            x = 2*(n-1)-x
        return x


    """if bound=='per':
        boundary = lambda x: np.mod(x,n)
    else:"""
    boundary = lambda x: [symmetrize(x[0],n), symmetrize(x[1],n)] # todo

    ##
    # For a given grid index |k|, and a given neighboring index k in \({1,2,3,4}\),
    # |Neigh(k,i)| gives the corresponding grid neigboring index.

    ind2sub1 = lambda k: [int( (k-np.fmod(k,n))/n ), np.fmod(k,n)]
    sub2ind1 = lambda u: int( u[0]*n + u[1] )
    Neigh = lambda k,i: sub2ind1(boundary(ind2sub1(k) + neigh[:,i]))
    extract   = lambda x,I: x[I]
    extract1d = lambda x,I: extract(x.flatten(),I)

    ##
    # Stack of starting points.

    nstart = pstart.shape[1]
    """I = list( np.zeros( (nstart, 1) ) )
    for i in np.arange(0, nstart):
        I[i] = int( sub2ind1(pstart[:,i]) )"""
    I = [sub2ind1(pstart)]

    ##
    # Initialize the distance to \(+\infty\), excepted for the boundary conditions.

    D = np.zeros( (n,n) ) + np.inf # current distance
    """for i in np.arange(0, nstart):
        D[int(pstart[0,i]),int(pstart[1,i])] = 0"""
    D[int(pstart[0,0]),int(pstart[1,0])] = 0


    ##
    # Initialize the state to 0 (unexplored), excepted for the boundary point to \(1\)
    # (front).

    S = np.zeros( (n,n) )
    S[int(pstart[0,0]),int(pstart[1,0])] = 1
    """for i in np.arange(0, nstart):
        S[int(pstart[0,i]),int(pstart[1,i])] = 1 # open"""

    ##
    # Run!

    iter = 0
    q = 100  # maximum number of saves
    Dsvg = np.zeros( (n,n,q) )
    Ssvg = np.zeros( (n,n,q) )
    while ( not(I==[]) & (iter<=niter) ):
        # print(not(I==[]) & (iter<=niter))
        iter = iter+1;
        # print(len(I))
        if iter==niter:
            break
        # pop from stack
        j = np.argsort( extract1d(D,I)  )
        if np.ndim(j)==0:
            j = [j]
        j = j[0]
        i = I[j]
        a = I.pop(j)
        # declare dead
        u = ind2sub1(i);
        S[u[0],u[1]] = -1
        # Make a list of neighbors that are not dead
        J = []
        for k in np.arange(0,4):
            j = Neigh(i,k)
            if extract1d(S,j)!=-1:
                # add to the list of point to update
                J.append(j)
                if extract1d(S,j)==0:
                    # add to the front
                    u = ind2sub1(j)
                    S[u[0],u[1]] = 1
                    I.append(j)
        # update neighbor values
        DNeigh = lambda D,k: extract1d(D,Neigh(j,k))
        for j in J:
            dx = min(DNeigh(D,0), DNeigh(D,1))
            dy = min(DNeigh(D,2), DNeigh(D,3))
            u = ind2sub1(j)
            w = extract1d(W,j);
            """if method=='dijstr':
                D[u[0],u[1]] = min(dx + w, dy + w)"""
            #else:
            Delta = 2*w - (dx-dy)**2
            if (Delta>=0):
                D[u[0],u[1]] = (dx + dy + np.sqrt(Delta))/ 2
            else:
                D[u[0],u[1]] = min(dx + w, dy + w)
        # svd
        #t = int(iter/svg_rate)
        """if (np.mod(iter,svg_rate)==0) & (t<q):
            Dsvg[:,:,t-1] = D
            Ssvg[:,:,t-1] = S"""
        
        if iter % 20000 == 0:
            plot(W,D,iter)

    #Dsvg = Dsvg[:,:,:t-1]
    #Ssvg = Ssvg[:,:,:t-1]
    return D;


def perform_fmm_2D(x0,W):
    """
    Compute the distance map to these starting point using the FM algorithm.
    """
    n = W.shape[0]
    pstart = np.transpose(np.array([x0]))
    D = perform_dijstra_fm(W, pstart, np.inf,'fm', 'sym',n*6)
    # display
    k = 8
    displ = lambda D: np.cos(2*np.pi*k*D/ max(D.flatten()))
    plt.clf
    #imageplot(D)
    c = plt.imshow(D)
    plt.colorbar(c)
    plt.set_cmap('jet')
    return D

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == '__main__':
    
    from imageio import imread
    from utils import perform_blurring, imageplot
    import cv2
    import os

    path = "../../data/training/images/"
 

    #directories = os.listdir(path)

    files = '27_training'

 
    img = cv2.cvtColor(cv2.imread('/Users/nicolasmakaroff/Documents/vessel_segmentation/data/training/images/' + files + '.tif'), cv2.COLOR_BGR2RGB)
    
    crop = img[584-564:584, 1:565]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    M = perform_blurring(gray,20)

    f1 = M - gray
    
    c = np.max(f1);
    epsilon = 1e-2;
    W = epsilon + np.abs(f1-c)
    imageplot(W)
    y = np.flip(np.array(plt.ginput(0), dtype=np.int16).flatten())
    plt.show()
    start_point = y.tolist()
    print(start_point)
    h, w = W.shape[:2]
    mask = create_circular_mask(h, w,radius=260)
    masked_img = W.copy()
    masked_img[~mask] = 255
    # Get brightness range - i.e. darkest and lightest pixels
    start_point = [230,70]
    D = perform_fmm_2D(start_point,masked_img)
    np.save('/Users/nicolasmakaroff/Documents/vessel_segmentation/code/fast_marching/training_fmm/%s'%files, D)
    