import sys
import matplotlib.pyplot as plt
sys.path.append("..")
import SNN

k = SNN.KittiData()
d = k.getdata("KITTI/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000000.bin")
print(d)
plt.imshow(d)
plt.show()