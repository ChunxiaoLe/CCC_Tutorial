#!/usr/bin/env 
from tkinter import colorchooser
from tkinter.tix import Tree
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as scio

def get_mcc_coord(fn):
        # Note: relative coord            #fn:8D5U5524.png
        with open( file_coorddinate+ fn.split('.')[0] +
                '_macbeth.txt', 'r') as f:
            lines = f.readlines()
            width, height = map(float, lines[0].split())
            scale_x = 1 / width
            scale_y = 1 / height
            lines = [lines[1], lines[2], lines[4], lines[3]]
            polygon = []
            for line in lines:
                line = line.strip().split()
                x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
                polygon.append((x, y))
            return np.array(polygon, dtype='float32')
def load_image_ori(fn):  
    file_name=file_raw+fn+".png"
    raw = np.array(cv2.imread(file_name, -1), dtype='float32')[:, :, :: -1]
    
    if fn.startswith('IMG'):
        black_point = 129
        camera = 'Canon5D'
    else:
        black_point = 0
        camera = 'Canon1D'
        
    img = np.maximum(raw - black_point, [0, 0, 0])
    img = img.astype(np.uint16)
    
    mask=np.ones_like(img)
    mcc_coord=get_mcc_coord(fn)
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)
    mask=cv2.fillPoly(mask, [polygon], (0) * 3)
    mask=mask[:, :, 0]
    h,w,c=np.shape(img)
    mask = mask.astype(np.bool_).reshape((h,w,1))
    return img,mask,camera
def load_image_resize(fn):  
    file_name=file_raw+fn+".png"
    raw = np.array(cv2.imread(file_name, -1), dtype='float32')[:, :, :: -1]
    
    if fn.startswith('IMG'):
        black_point = 129
        camera = 'Canon5D'
    else:
        black_point = 0
        camera = 'Canon1D'
        
    img = np.maximum(raw - black_point, [0, 0, 0])
    img = img.astype(np.uint16)

    mask=np.ones_like(img)
    mcc_coord=get_mcc_coord(fn)
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)
    mask=cv2.fillPoly(mask, [polygon], (0) * 3)
    mask=mask[:, :, 0]
    
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) * 4
    mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5) * 4
    
    h,w,c=np.shape(img)
    mask = mask.astype(np.bool_).reshape((h,w,1))
    return img,mask,camera


"""
Color Checker Data preprocessing:
1: Download the dataset on: https://www2.cs.sfu.ca/~colour/data/shi_gehler/
2: Store the download file in: /dataset/colorconstancy/colorchecker2010/
3: Set the path of the output file, such as: /home/***/data/CC_full_size or /home/***/data/CC_resize
4: Run the following code
"""
ColorChecker=True
if ColorChecker:
    file_illum="/dataset/colorconstancy/colorchecker2010/real_illum_568..mat"
    file_name="/dataset/colorconstancy/colorchecker2010/img.txt"
    file_coorddinate='/dataset/colorconstancy/colorchecker2010/coordinates/'
    file_raw="/dataset/colorconstancy/colorchecker2010/"
    output_path_full="/home/***/data/CC_full_size"
    output_path_resize="/home/***/data/CC_resize"



    gt=scio.loadmat(file_illum)["real_rgb"]
    img_gt = gt / np.linalg.norm(gt, ord=2, axis=1).reshape(-1, 1)
    img_list = []
    with open(file_name, "r") as f:
        for line in f:  
            line = line.rstrip()
            img_list.append(line)

    print(len(img_list),len(img_gt))

    output_path=output_path_full
    for idx in range(len(img_list)):
        RAW=img_list[idx]
        img,mask,camera=load_image_ori(RAW)
        np.save('{}/{}.npy'.format(output_path, img_list[idx]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_list[idx]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_list[idx]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_list[idx]), img_gt[idx])
        print(output_path+"/"+img_list[idx])

    output_path=output_path_resize
    for idx in range(len(img_list)):
        RAW=img_list[idx]
        img,mask,camera=load_image_resize(RAW)
        np.save('{}/{}.npy'.format(output_path, img_list[idx]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_list[idx]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_list[idx]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_list[idx]), img_gt[idx])
        print(output_path+"/"+img_list[idx])






"""
Cube+ :Data preprocessing
1: Download the dataset on: https://ipg.fer.hr/ipg/resources/color_constancy
2: Store the download file in: /dataset/colorconstancy/Cube/
3: Set the path of the output file, such as: /home/***/data/Cube_full_size or /home/***/data/Cube_resize
4: Run the following code
"""
Cube=True
if Cube:
    file_illum_path="/dataset/colorconstancy/Cube/"
    file_name_path="/dataset/colorconstancy/Cube/img.txt"
    file_raw="/dataset/colorconstancy/Cube/"
    output_path_full="/home/***/data/Cube_full_size/"
    output_path_resize="/home/***/data/Cube_resize/"




    img_gt = []
    with open(file_illum_path + "cube+_gt.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            item = list(map(float, line.split(' ')))
            img_gt.append(item)
    img_gt = np.array(img_gt)
    img_gt = img_gt / np.linalg.norm(img_gt, ord=2, axis=1).reshape(-1, 1)
    print(img_gt.shape)



    img_list = []
    with open(file_name_path, "r") as f:
        for line in f:  
            line = line.rstrip()
            img_list.append(line)
    print(len(img_list))



    output_path=output_path_full
    number=len(img_list)
    for idx in range(number):
        file_name=file_raw+img_list[idx]
        print(idx,file_name)
        img=np.array(cv2.imread(file_name, -1), dtype='float32')[:, :, :: -1]
        mask=np.ones_like(img)[:,:,0]
        mask[1050:, 2050:] = 0
        gt=img_gt[idx]
        saturationLevel = np.max(img) - 2
        
        blackLevel = 2048
        img=img-blackLevel
        
        img[img<0] = 0
        img[img > saturationLevel - blackLevel] = saturationLevel - blackLevel
        h,w,c = img.shape
        mask = mask.astype(np.bool_).reshape((h,w,1))
        img = img.astype(np.uint16)
        
        np.save('{}/{}.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), img)
        np.save('{}/{}_mask.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), 'Canon550D')
        np.save('{}/{}_gt.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), gt)




        output_path=output_path_resize
        number=len(img_list)
        for idx in range(number):
            file_name=file_raw+img_list[idx]
            print(idx,file_name)
            img=np.array(cv2.imread(file_name, -1), dtype='float32')[:, :, :: -1]
            mask=np.ones_like(img)[:,:,0]
            mask[1050:, 2050:] = 0
            gt=img_gt[idx]
            saturationLevel = np.max(img) - 2
            
            blackLevel = 2048
            img=img-blackLevel
            
            img[img<0] = 0
            img[img > saturationLevel - blackLevel] = saturationLevel - blackLevel
            
            
            
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) * 4
            mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5) * 4
            
            h,w,c = img.shape
            mask = mask.astype(np.bool_).reshape((h,w,1))
            img = img.astype(np.uint16)
            
            np.save('{}/{}.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), img)
            np.save('{}/{}_mask.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), mask)
            np.save('{}/{}_camera.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), 'Canon550D')
            np.save('{}/{}_gt.npy'.format(output_path, file_name.split('/')[-1].split('.')[0]), gt)




"""
NUS :Data preprocessing
1: Download the dataset on: https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html
2: Store the download file in: /dataset/colorconstancy/NUS/
3: Set the path of the output file, such as: /home/***/data/NUS_full_size or /home/***/data/NUS_resize
4: Run the following code
"""
NUS=True 
if NUS:
    data_dir="/dataset/colorconstancy/NUS/"
    output_path_full= '/home/***/data/NUS_full_size/'
    output_path_resize='/home/***/data/NUS_resize/'


    img_list = []
    k = 0

    camera_list = ["Canon1DsMkIII","Canon600D","FujifilmXM1","NikonD5200","OlympusEPL6","PanasonicGX1","SamsungNX2000","SonyA57"]
    for camera in camera_list:
        mat = scio.loadmat(data_dir + camera + '/' + camera + '_gt.mat')
        print(camera)
        for i in range(mat['all_image_names'].shape[0]):
            img_p = data_dir + camera + '/' + mat['all_image_names'][i][0][0] + ".PNG"
            img_list.append({'imgpath':img_p,                                 'saturation_level':mat['saturation_level'][0][0],                                 'camera':camera,                                'darkness_level':mat['darkness_level'][0][0],                                 'gt':mat['groundtruth_illuminants'][i],                                 'mcc':mat['CC_coords'][i],                                 'k':k})
            k+=1



    output_path = output_path_full
    #SATURATION_SCALE = 0.95

    for idx in range(len(img_list)):
        img_path = img_list[idx]['imgpath']
        saturationLevel = img_list[idx]['saturation_level']
        darkness_level = img_list[idx]['darkness_level']
        gt = img_list[idx]['gt']
        mcc = img_list[idx]['mcc']
        camera = img_list[idx]['camera']
        
        
        img = np.array(cv2.imread(img_path, -1), dtype='float32')[:, :, :: -1]
        img=img-darkness_level
        img[img<0] = 0
        h,w,c = img.shape
        sat = (saturationLevel - darkness_level)# * SATURATION_SCALE
        img[img > sat] = sat
        img = np.clip(img, 0, 65535)
        
        
        coor = [[mcc[2], mcc[0]],[mcc[2], mcc[1]],[mcc[3], mcc[1]],[mcc[3], mcc[0]]]
        coor = np.array(coor).astype(np.int32)
        mask = np.ones((h, w)).astype(np.float64)
        mask = cv2.fillPoly(mask, [coor], (0, 0, 0))
        
        
        h,w,c = img.shape
        mask = mask.astype(np.bool_).reshape((h,w,1))
        img = img.astype(np.uint16)
        
        print(idx, img_path)
        np.save('{}/{}.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), gt)
        



    output_path =output_path_resize
    #SATURATION_SCALE = 0.95

    for idx in range(len(img_list)):
        img_path = img_list[idx]['imgpath']
        saturationLevel = img_list[idx]['saturation_level']
        darkness_level = img_list[idx]['darkness_level']
        gt = img_list[idx]['gt']
        mcc = img_list[idx]['mcc']
        camera = img_list[idx]['camera']
        
        
        img = np.array(cv2.imread(img_path, -1), dtype='float32')[:, :, :: -1]
        img=img-darkness_level
        img[img<0] = 0
        h,w,c = img.shape
        sat = (saturationLevel - darkness_level)# * SATURATION_SCALE
        img[img > sat] = sat
        img = np.clip(img, 0, 65535)
        
        
        coor = [[mcc[2], mcc[0]],[mcc[2], mcc[1]],[mcc[3], mcc[1]],[mcc[3], mcc[0]]]
        coor = np.array(coor).astype(np.int32)
        mask = np.ones((h, w)).astype(np.float64)
        mask = cv2.fillPoly(mask, [coor], (0, 0, 0))
        

        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) * 4
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5) * 4
        
        h,w,c = img.shape
        mask = mask.astype(np.bool_).reshape((h,w,1))
        img = img.astype(np.uint16)
        
        print(idx, img_path)
        np.save('{}/{}.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), img)
        np.save('{}/{}_mask.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), mask)
        np.save('{}/{}_camera.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), camera)
        np.save('{}/{}_gt.npy'.format(output_path, img_path.split('/')[-1].split('.')[0]), gt)

    

