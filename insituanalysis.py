#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import numpy as np
import cv2
import ncempy.io as nc
import matplotlib.pyplot as plt

crop_size = 640

def load_insitu(path):
    data = list()
    files = glob.glob(path+'/**/*.dm4', recursive = True)
    files.sort()
    n = 0
    for file in files:
        image = nc.read(file)
        data.append(image['data'])
        if n == 0:
            pixel_size = image['pixelSize']
            n = 1
    return(np.array(data), pixel_size[0], files)

def gaussian(x, m, s):
    return np.exp(-0.5*((x-m)/s)**2)
    
def Gaussian_adjust(image):
    im = image.copy()
    m = np.mean(im.ravel())
    s = np.std(im.ravel())
    g = gaussian(im, m, s)
    im[np.logical_and(g<0.001, im>m)] = m+3.5*s
    return im

def rescale(im): 
    m = np.mean(im)
    s = np.std(im)
    return (im - m) / s  

def norm(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im)) 

def dfourier(im):
    dft = cv2.dft(np.float32(im),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return(dft_shift, magnitude_spectrum)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    all_r = np.arange(r.min(), r.max()+1, 1)
    non_nan = radialprofile==radialprofile
    
    return radialprofile[non_nan], all_r[non_nan]

def angular_profile(data, center, r_range):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan((y - center[1])/(x - center[0]+1e-10))/np.pi*180+90
    theta = theta.astype(np.int)
    
    theta_flat = theta.ravel()
    data_flat = data.ravel()
    r_flat = r.ravel()
    
    roi = np.logical_and(r_flat>r_range[0], r_flat<r_range[1])

    tbin = np.bincount(theta_flat[roi], data_flat[roi])
    ntheta = np.bincount(theta_flat[roi])
    angularprofile = tbin / ntheta
    
    all_theta = np.arange(theta_flat[roi].min(), theta_flat[roi].max()+1, 1)
    non_nan = angularprofile==angularprofile
    
    return angularprofile[non_nan], all_theta[non_nan]

def gaussian2d(size, mu, sigma):
    x, y = np.meshgrid(np.arange(0, size[0], 1), np.arange(0, size[1], 1))
    return np.exp(-((x-mu[0])**2/(2*sigma[0]**2)+(y-mu[1])**2/(2*sigma[1]**2)))

def simulate_image(period, rot=0, defect=False):
    #period = 24.7 #20.5 for TEM
    locs_x = np.arange(130, 510, period)
    locs_xx, locs_yy = np.meshgrid(locs_x, locs_x)
    locs_xx = locs_xx+0.1*(np.random.random(locs_xx.shape)-0.5)
    locs_yy = locs_yy+0.1*(np.random.random(locs_yy.shape)-0.5)
    locs = np.array([locs_xx.flatten(), locs_yy.flatten()]).transpose()
    
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    locs = np.round(np.dot(locs-320, rot_mat)+320).astype('int')
    simulated_image = np.zeros((640,640))+0.2
    kernel_center = gaussian2d([64, 64], [32, 32], [7.5*1.17, 7.5*1.17])*0.75
    if defect:
        if defect == 'mid':
            defect = np.round(locs.shape[0]/2).astype('int')
        idx = np.ones(locs.shape[0]).astype('bool')
        idx[defect] = False
        locs = locs[idx, :]
    for i in range(locs.shape[0]):
            kernel = np.zeros((640,640))
            kernel[locs[i, 0]-32:locs[i, 0]+32, locs[i, 1]-32:locs[i, 1]+32] = kernel_center
            simulated_image += kernel
    return(1-simulated_image)

def get_center(im):
    bad = True
    cnt = 0
    while bad and cnt<10:
        centerx = int(input('Input the row index of the center:'))
        centery = int(input('Input the column index of the center:'))
        plt.figure(figsize=(10,10))
        plt.imshow(Gaussian_adjust(im[centerx-crop_size:centerx+crop_size, centery-crop_size:centery+crop_size]), cmap='gray')
        plt.show()
        check = input('Does this look good (y/n):')
        while check!='y' and check!='n':
            check = input('Unrecognized input, please try again:')
        if check=='y':
            bad = False
        else:
            bad = True
        cnt+=1
    if cnt>=10:
        print('Too many attempts, please restart')
    return([centerx, centery])