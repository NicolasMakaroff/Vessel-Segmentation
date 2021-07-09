import numpy as np
from utils import imageplot

def grad(f):
    """
        Compute a finite difference approximation of the gradient of a 2D image, assuming periodic BC.
    """
    S = f.shape;
#   g = np.zeros([n[0], n[1], 2]);
    s0 = np.concatenate( (np.arange(1,S[0]),[0]) )
    s1 = np.concatenate( (np.arange(1,S[1]),[0]) )
    g = np.dstack( (f[s0,:] - f, f[:,s1] - f))
    return g

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def geodesicPath(tau,x0,x1,G):
    """
    Perform the full geodesic path extraction by iterating the gradient
    descent. You must be very careful when the path become close to
    $x_0$, because the distance function is not differentiable at this
    point. You must stop the iteration when the path is close to $x_0$.
    """
    n = G.shape[0]
    Geval = lambda G,x: bilinear_interpolate(G[:,:,0], np.imag(x), np.real(x) ) + 1j * bilinear_interpolate(G[:,:,1],np.imag(x), np.real(x))
    niter = 1.5*n/tau;
    # init gamma
    gamma = [x1]
    xtgt = x0[0] + 1j*x0[1]
    for i in np.arange(0,niter):
        g = Geval(G, gamma[-1] )
        gamma.append( gamma[-1] - tau*g )
        if np.abs(gamma[-1]-xtgt).all()<1:
            break
    gamma.append( xtgt )
    return gamma

if __name__ == '__main__':
    from imageio import imread
    from utils import perform_blurring, imageplot
    import cv2
    import matplotlib.pyplot as plt
    
    img = cv2.cvtColor(cv2.imread('../../data/40_training.tif'), cv2.COLOR_BGR2RGB)
    
    crop = img[584-564:584, 1:565]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    M = perform_blurring(gray,20)

    f1 = M - gray
    
    c = np.max(f1);
    epsilon = 1e-2;
    W = epsilon + np.abs(f1-c)
    D = np.load('fmm.npy')

    G0 = grad(D)

    n = G0.shape[0]

    d = np.sqrt(np.sum(G0**2, axis=2))

    U = np.zeros((n,n,2))
    U[:,:,0] = d
    U[:,:,1] = d
    G = G0 / U
    gamma = geodesicPath(.8,[250,500],140 + 1j*60,G)
    imageplot(W) 
    plt.set_cmap('gray')
    x1 = 140 + 1j*60
    x0 = [250,500]

    h = plt.plot(np.imag(gamma), np.real(gamma), '.b', linewidth=2)
    h = plt.plot(x0[1], x0[0], '.r', markersize=20)
    h = plt.plot(np.imag(x1), np.real(x1), '.g', markersize=20)
    plt.savefig('geod.png')