import torch as t
from datasets.rail_defect import RDDataset
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import  numpy as np
from datetime import datetime
import cv2
import torchvision

rootpath = '/home/wjy/rail_362/Dataset/'
TestDatasets = RDDataset(rootpath, 'test')
test_dataloader = DataLoader(TestDatasets, batch_size=1, shuffle=False, num_workers=4)
# **************************************************
from model.RD_T import *
# from model.REDENet.REDENet import *
# from model.REDENet.REDENet_backbone import *
# net = AAANet()
# net = AAANet_S([32, 64, 160, 256], 'segformer_b0')
# net = AAANet_S([64, 128, 256, 512], 'shunted_t')
net = AAANet_T([64, 128, 256, 512], 'shunted_b')
net.load_state_dict(t.load('/home/wjy/RAIL_DEFECT_DETECTION/kdpth/pretrained_t.pth'))
# a = '/media/wjy/6D8C8BC057829E28/RGBT-EvaluationTools/SalMap/'
a = '/home/wjy/RAIL_DEFECT_DETECTION/PIC/'
b = 'odepth1'
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'
vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e
# ***************************************************
path1 = vt800

isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		out = net(image, depth)

		out = torch.softmax(out[sup])
		out = torch.sigmoid(out[5])

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()
		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')

		# out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
		# out_img = out.cpu().detach().numpy()
		# out_img = np.max(out_img, axis=1).reshape(320, 320)
		# out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
		# out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
		# cv2.imwrite(path1 + name + '.png', out_img)
		# print(path1 + name + '.png')


##########################################################################################
# path2 = vt1000
# isExist = os.path.exists(vt1000)
# if not isExist:
# 	os.makedirs(vt1000)
# else:
# 	print('path2 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader2):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5 = net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
# 		plt.imsave(path2 + name + '.png', arr=out_img, cmap='gray')
# 		print(path2 + name + '.png')
# 	cur_time = datetime.now()
#######################################################################################################
#
# path3 = vt5000
# isExist = os.path.exists(vt5000)
# if not isExist:
# 	os.makedirs(vt5000)
# else:
# 	print('path3 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader3):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5= net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
#
# 		plt.imsave(path3 + name + '.png', arr=out_img, cmap='gray')
# 		print(path3 + name + '.png')
#
# 	cur_time = datetime.now()
#   TIANCAIDAOCIYIYOU








