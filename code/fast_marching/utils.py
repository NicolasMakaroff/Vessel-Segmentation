import pylab 
import numpy as np
import matplotlib.pyplot as plt

def imageplot(f, str='', sbpt=[]):
    """
        Use nearest neighbor interpolation for the display.
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    plt.axis('off')
    if str != '':
        plt.title(str)
        
def perform_blurring(f, sigma):

    """ gaussian_blur - gaussian blurs an image
    %
    %   M = perform_blurring(M, sigma, options);
    %
    %   M is the original data
    %   sigma is the std of the Gaussian blur (in pixels)
    %
    %   Copyright (c) 2007 Gabriel Peyre
    """
    if sigma<=0:
        return;
    n = max(f.shape);
    t = np.concatenate( (np.arange(0,n/2+1), np.arange(-n/2,-1)) )
    [Y,X] = np.meshgrid(t,t)
    h = np.exp( -(X**2+Y**2)/(2.0*float(sigma)**2) )
    h = h/np.sum(h)
    return np.real( pylab.ifft2(pylab.fft2(f) * pylab.fft2(h)) )
