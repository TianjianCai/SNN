import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def chartesian_to_spherical_conversion(xyz):
    rtp = np.zeros_like(xyz)
    xsq = xyz[:,0]**2
    ysq = xyz[:,1]**2
    zsq = xyz[:,2]**2
    xy = np.sqrt(xsq+ysq)
    rtp[:,0] = np.sqrt(xsq + ysq + zsq)
    rtp[:,1] = np.arctan2(xy,xyz[:,2])
    rtp[:,2] = np.arctan2(xyz[:,1],xyz[:,0])
    return rtp


def rtp_to_xy(rtp):
    xy = np.zeros_like(rtp)
    xy[:,0] = (-rtp[:,2]+3.14)*180/3.14
    xy[:,1] = (rtp[:,1])*180/3.14
    xy[:,2] = rtp[:,0]
    return xy


def xy_to_map(xy):
    scale = 2
    map = 100*np.ones([30*scale,360*scale])
    for element in xy:
        if map[int((element[1]-85)*scale-1),int(element[0]*scale-1)] > element[2]:
            map[int((element[1]-85)*scale-1),int(element[0]*scale-1)] = element[2]
    return map

path = os.listdir(os.getcwd()+"/KITTI_data/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/")
print(path)
scan = np.fromfile(os.getcwd()+"/KITTI_data/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/" + path[30], dtype=np.float32)
scan = np.reshape(scan,[-1,4])
rtp = chartesian_to_spherical_conversion(scan)

xy = rtp_to_xy(rtp)
map = xy_to_map(xy)

f2 = plt.figure(figsize=(20,4))
ax2 = f2.add_subplot(111)
ax2.imshow(map)


plt.show()



