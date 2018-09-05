import os
import pykitti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

basedir = os.getcwd() + '\\KITTI_data'
date = '2011_09_26'
drive = '0005'

dataset = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))


# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

# Grab some data

velo0 = dataset.get_velo(0)
velo1 = dataset.get_velo(1)
velo2 = dataset.get_velo(9)




f1 = plt.figure()
ax2 = f1.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range0 = range(0, velo0.shape[0], 100)
ax2.scatter(velo0[velo_range0, 0],
            velo0[velo_range0, 1],
            velo0[velo_range0, 2],
            c=velo0[velo_range0, 3],
            cmap='gray')

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range1 = range(0, velo1.shape[0], 100)
ax2.scatter(velo1[velo_range1, 0],
            velo1[velo_range1, 1],
            velo1[velo_range1, 2],
            c=velo1[velo_range1, 3],
            cmap='gray')

f3 = plt.figure()
ax2 = f3.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range2 = range(0, velo2.shape[0], 3)
ax2.scatter(velo2[velo_range2, 0],
            velo2[velo_range2, 1],
            velo2[velo_range2, 2],
            c=velo2[velo_range2, 3],
            cmap='gray',s=0.1)


plt.show()