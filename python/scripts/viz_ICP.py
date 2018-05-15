import os,sys
sys.path.insert(0, '../')

import numpy as np
from scipy import misc
from common import common
from common import util
from director import vtkAll as vtk
import yaml
import time
from director import vtkNumpy as vnp
from director import filterUtils
from RGBDCNN import network
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import os
import glob

path = '/media/drc/DATA/chris_labelfusion/logs1/'
files = os.listdir(path)
real = glob.glob(path+"*realdepth*")
mysim = glob.glob(path+"*depthsim*")
othersim = glob.glob(path+"*kunidepth*")
sim = glob.glob(path+"*_sim*")
print sim

for i in [real,mysim,othersim,sim]:
  edistances = []
  rdistances = []
  for j in i:
    with open(j, 'r') as stream:
      dic = yaml.load(stream)
      ground_truth_pose, post_icp_pose = (dic['ground_truth_pose'], dic['post_icp_pose'])
      for key in ground_truth_pose.keys():
        try:
          pos1 = np.array(ground_truth_pose[key][0])
          pos2 = np.array(post_icp_pose[key][0])
          quat1 = np.array(ground_truth_pose[key][1])
          quat2 = np.array(post_icp_pose[key][1])
          edistances.append(np.linalg.norm(pos1-pos2))
          rdistances.append(np.arccos(2*np.dot(quat1,quat2)**2-1))
        except:
          pass

  print "mean ecludian distance ",np.mean(edistances)
  print "variance ecludian distance ", np.var(edistances)
  print "mean rotational distance ", np.mean(rdistances)
  print "variance rotational distance " ,np.var(rdistances)
  print("\n")

