import cv2
import matplotlib.pyplot as plt
import os
from scipy import misc
import numpy as np
import data
import network
import glob
import scipy
import time

'''
This file contains code to visualize a simulation of an RGBD stream 
with the CNN postprocessing images to add droopout noise
'''

#viz method for prediction with generator
def viz_predicted_depth(path,model_path,sleep =.1,filter_files= None,img_height=480,img_width=640,save_dir = None): #add file filter for specific logs
	model = network.load_trained_model(weights_path = model_path)
	samples = data.gen_samples(path,False,filter_files=filter_files)
	
	print "generarting samples"
	train = data.generate_data_custom_depth(samples,batch_size = 1)
	
	stack = train.next()
	depth = np.reshape(stack[1],(img_height,img_width))
	gtdepth = np.reshape(stack[0],(img_height,img_width))

	threshold = .3
	predicted_prob_map = model.predict_on_batch(stack[0])
	network.apply_mask(predicted_prob_map,gtdepth,threshold)

	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	#ax3 = plt.subplot(1,3,3)

	im1 = ax1.imshow(depth)
	im2 = ax2.imshow(gtdepth*3500)
	#im3 = ax3.imshow(depth)

	plt.ion()
	
	int = 0
	while True:
		stack = train.next()
		depth = np.reshape(stack[1],(img_height,img_width))
		gtdepth = np.reshape(stack[0],(img_height,img_width))


		predicted_prob_map = model.predict_on_batch(stack[0])
		network.apply_mask(predicted_prob_map,gtdepth,threshold)
		im1.set_data(depth)
		im2.set_data(gtdepth*3500)
		#im3.set_data(depth)

		if save_dir:
			misc.imsave(save_dir)
		plt.pause(sleep)

	plt.ioff() # due to infinite loop, this gets never called.
	plt.show()


#viz method for prediction from directory
def viz_predicted_depth1(path,model_path,sleep =.1,filter_files= None,img_height=480,img_width=640,save_dir = None,viz=True): #add file filter for specific logs
	model = network.load_trained_model(weights_path = model_path)
	samples = gen_samples(path,False,filter_files=filter_files)
	stack = np.zeros((1,img_height,img_width,1))

	max_depth = 3500.

	#threshold where NDP probabilities greater than it will be classified as NDP, 
	#NDP probabilities lower will be instiatiated with correlated noise process
	threshold = .5

	ax1 = plt.subplot(1,4,1)
	ax2 = plt.subplot(1,4,2)
	ax3 = plt.subplot(1,4,3)
	ax4 = plt.subplot(1,4,4)

	im1 = ax1.imshow(misc.imread(samples[0][1]))
	im2 = ax2.imshow(misc.imread(samples[0][0]))
	im3 = ax3.imshow(misc.imread(samples[0][0])/max_depth)
	im4 = ax4.imshow(misc.imread(samples[0][2]))
	plt.ion()

	for i in range(len(samples)):
		#read images
		rgb = misc.imread(samples[i][2])
		depth = misc.imread(samples[i][1])
		gtdepth = misc.imread(samples[i][0])/max_depth
		stack[0,:,:,0] = gtdepth
		gt_copy = np.copy(gtdepth)

		#predict NDP
		predicted_prob_map = model.predict_on_batch(stack)
		
		#apply NDP to 'perfect sim'
		network.apply_mask(predicted_prob_map,gtdepth,threshold)

		im1.set_data(depth)
		im2.set_data(gtdepth*max_depth)
		im3.set_data(gt_copy)
		im4.set_data(rgb)

		if save_dir:

			scipy.misc.toimage(rgb).save(save_dir+str(i)+"rgb.png")
			scipy.misc.toimage(depth, cmin=0, cmax=max_depth,mode = "I").save(save_dir+str(i)+"depth.png")
			scipy.misc.toimage(gt_copy*max_depth, cmin=0, cmax=max_depth,mode = "I").save(save_dir+str(i)+"gtdepth.png")
			scipy.misc.toimage(gtdepth*max_depth, cmin=0, cmax=max_depth,mode = "I").save(save_dir+str(i)+"predicted_depth.png")
		plt.pause(sleep)

	plt.ioff() # due to infinite loop, this gets never called.
	plt.show()

#method for organizing files to process
def gen_samples(directory,shuffle = True,filter_files=None):
    samples = []
    dirs = os.listdir(directory)
    for i in dirs:
		if filter_files and i in filter_files:
			path = os.path.join(directory, i)+"/"
			if os.access(path, os.R_OK):
				gt_depth = sorted(glob.glob(path+"*_depth_*"))
				depth = sorted(glob.glob(path+"*_depth.png"))
				rgb = sorted(glob.glob(path+"*rgb.png"))
				samples.extend(zip(gt_depth,depth,rgb))
    if shuffle:
        random.shuffle(samples)
    return samples 

def show_prob_map_dist(img):
	plt.figure()
	plt.hist(img.ravel(), bins=256)
	plt.show()

if __name__ == '__main__':
	save_dir = "/media/drc/DATA/chris_labelfusion/RGBDCNNTest/"
	viz_predicted_depth1(sleep =.05,filter_files = " 2017-06-16-19",path = "/media/drc/DATA/chris_labelfusion/RGBDCNN/",model_path = "../models/net_depth_seg_v1.hdf5")#,save_dir = save_dir)