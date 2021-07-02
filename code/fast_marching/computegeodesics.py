import numpy as np

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
