import os
import cv2
import json
import torch 
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

COCO_DICT=['images','annotations','categories']

IMAGES_DICT=['file_name','height','width','id']
## {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
## 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
## 'flickr_url': '', 'id': 516316}

ANNOTATIONS_DICT=['image_id','iscrowd','area','bbox','category_id','id','segmentation']
## {'segmentation': [[]],
## 'area': 58488.148799999995, 'iscrowd': 0, 
## 'image_id': 170893, 'bbox': [270.55, 80.55, 367.51, 393.7],
## 'category_id': 18, 'id': 9940}

CATEGORIES_DICT=['id','name']
## {'supercategory': 'person', 'id': 1, 'name': 'person'}
## {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}

DIOR_CATEGORIES=['ship']
num_dict={x:0 for x in DIOR_CATEGORIES}

parser=argparse.ArgumentParser(description='2COCO')
parser.add_argument('--image_path',type=str,default='./RGB/',help='config file')
parser.add_argument('--annotation_path',type=str,default='./xml/',help='config file')
parser.add_argument('--save',type=str,default='./test1.json',help='config file')

args=parser.parse_args()

def load_json(path):
	with open(path,'r') as f:
		json_dict=json.load(f)
		for i in json_dict:
			print(i)
		print(json_dict['annotations'])

def save_json(dict,path):
	print('SAVE_JSON...')
	with open(path,'w') as f:
		json.dump(dict,f)
	print('SUCCESSFUL_SAVE_JSON:',path)

def load_image(path):
	img=cv2.imread(path)
	return img.shape[0],img.shape[1]

def generate_categories_dict(category):
	print('GENERATE_CATEGORIES_DICT...')
	return [{CATEGORIES_DICT[0]:category.index(x)+1,CATEGORIES_DICT[1]:x} for x in category]

def generate_images_dict(imagelist,image_path):
	print('GENERATE_IMAGES_DICT...')
	images_dict=[]
	with tqdm(total=len(imagelist)) as load_bar:
		for x in imagelist:
			dict={IMAGES_DICT[0]:x,IMAGES_DICT[1]:load_image(image_path+x)[0],\
					IMAGES_DICT[2]:load_image(image_path+x)[1],IMAGES_DICT[3]:int(x.split('.')[0])}
			load_bar.update(1)
			images_dict.append(dict)
	return images_dict
	# return [{IMAGES_DICT[0]:x,IMAGES_DICT[1]:load_image(image_path+x)[0],\
	# 				IMAGES_DICT[2]:load_image(image_path+x)[1],IMAGES_DICT[3]:imagelist.index(x)+start_image_id} for x in imagelist]

def  DIOR_Dataset(image_path,annotation_path,start_image_id=0,start_id=0):
	num=0
	categories_dict=generate_categories_dict(DIOR_CATEGORIES)

	imgname=os.listdir(image_path)
	images_dict=generate_images_dict(imgname,image_path)

	print('GENERATE_ANNOTATIONS_DICT...')
	annotations_dict=[]
	id=start_id

	for i in images_dict:
		# print(i)
		image_id=i['id']
		image_name=i['file_name']
		annotation_xml=annotation_path+image_name.split('.')[0]+'.xml'

		tree=ET.parse(annotation_xml)
		root=tree.getroot()

		for j in root.findall('objects'):
			for m in j.findall('object'):
				category=m.find('possibleresult').find('name').text
				category_id=DIOR_CATEGORIES.index(category)+1
				num_dict[category]=num_dict[category]+1
				x=[]
				y=[]
				for k in m.find('points'):
					x.append(float(k.text.split(',')[0]))
					y.append(float(k.text.split(',')[1]))
				x_min=min(x)#max(min(x),0)
				y_min=min(y)#max(min(y),0)
				x_max=max(x)#min(max(x),i['width'])
				y_max=max(y)#min(max(y),i['height'])
				w=x_max-x_min
				h=y_max-y_min
				# x=torch.Tensor(x[:4])
				# y=torch.Tensor(y[:4])
				# a1=y==y_min
				# a2=x==x_max
				# a3=y==y_max
				# a4=x==x_min
				# d=[None,None,None,None]
				# if a1.sum()==2 and a2.sum()==2 and a3.sum()==2 and a4.sum()==2:
				# 	d=[0,0,0,0]
				# 	a1[a1]=False
				# 	a2[a2]=False
				# 	a3[a3]=False
				# 	a4[a4]=False
				# for mm in range(4):
				# 	if a3.sum()==1 and d[2]==None:
				# 		d[2]=((x_max-x[a3])/w).item()
				# 		a1[a3]=False
				# 		a2[a3]=False
				# 		a4[a3]=False
				# 		a3[a3]=False
				# 	if a2.sum()==1 and d[1]==None:
				# 		d[1]=((y[a2]-y_min)/h).item()
				# 		a1[a2]=False
				# 		a3[a2]=False
				# 		a4[a2]=False
				# 		a2[a2]=False
				# 	if a1.sum()==1 and d[0]==None:
				# 		d[0]=((x[a1]-x_min)/w).item()
				# 		a2[a1]=False
				# 		a3[a1]=False
				# 		a4[a1]=False
				# 		a1[a1]=False
				# 	if a4.sum()==1 and d[3]==None:
				# 		d[3]=((y_max-y[a4])/h).item()
				# 		a1[a4]=False
				# 		a2[a4]=False
				# 		a3[a4]=False
				# 		a4[a4]=False
				# bbox=[x_min,y_min,w,h,d[0],d[1],d[2],d[3]]
				bbox=[x_min,y_min,w,h]
				segmentation=[[x_min,y_min,x_min,y_max,x_max,y_max,x_max,y_min]]
				dict={'image_id':image_id,'iscrowd':0,'bbox':bbox,'area':w*h,'category_id':category_id,'id':id,'segmentation':segmentation,'ignore':0}
				annotations_dict.append(dict)
				id=id+1
	print('SUCCESSFUL_GENERATE_DIOR_JSON')

	return {COCO_DICT[0]:images_dict,COCO_DICT[1]:annotations_dict,COCO_DICT[2]:categories_dict}

if __name__=='__main__':

	#dataset=args.dataset
	save=args.save
	image_path=args.image_path
	annotation_path=args.annotation_path

	json_dict=DIOR_Dataset(image_path,annotation_path,0)
	save_json(json_dict,save)