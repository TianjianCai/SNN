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

path = os.listdir(os.getcwd()+"/KITTI_data/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/")
print(path)
scan = np.fromfile(os.getcwd()+"/KITTI_data/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/" + path[120], dtype=np.float32)
scan = np.reshape(scan,[-1,4])
rtp = chartesian_to_spherical_conversion(scan)

f1 = plt.figure(figsize=(20,4))
ax1 = f1.add_subplot(111)
# Plot every 100th point so things don't get too bogged down
velo_range2 = range(0, rtp.shape[0], 1)
ax1.scatter(-rtp[velo_range2, 2],
            -rtp[velo_range2, 1],
            c=rtp[velo_range2, 0],
            s=1)



plt.show()



