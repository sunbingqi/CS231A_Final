import numpy as np
import pptk

SEQ_NUM = '00' # sequence number of KITTI dataset
SIZE = 100 # number of frames used for 3D rendering

data = np.loadtxt(SEQ_NUM + '/frames/frame0.txt')
for i in range(1, SIZE):
    cur = np.loadtxt(SEQ_NUM + '/frames/frame' + str(i) + '.txt')
    data = np.vstack((data, cur))

pos = data[:, :3] # get positions
color = data[:, 3:] / 255 # get corresponding colors

# view semantic mapping
v = pptk.viewer(pos)
v.attributes(color)
v.set(point_size=0.05)
v.wait()