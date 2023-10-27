################################################################
#Step 2 EDA- Geometric transformation analysis of images
import cv2
import matplotlib.pyplot as plt

img_path_1 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\C130\\0a317b1280597e9e84d2891ee88a8bb2_1.jpg'
img_1 = cv2.imread(img_path_1)
img_path_2 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\F18\\4c4a9626fea6f02ae260c2300e87d83e_7.jpg'
img_2 = cv2.imread(img_path_2)

#Basic image manipulation (rotating/flipping/transpose)
flip_img_v1=cv2.flip(img_1,0) # vertical flip
flip_img_v2=cv2.flip(img_2,0) # vertical flip
#horizontal flip
flip_img_h1=cv2.flip(img_1,1) # horizontal flip
flip_img_h2=cv2.flip(img_2,1) # horizontal flip
#transpose
transp_img_1=cv2.transpose(img_1,1) # transpose
transp_img_2=cv2.transpose(img_2,1) # transpose

plt.figure(figsize=(10,10))
plt.subplot(321)
plt.imshow(flip_img_v1),plt.title('Vertical flipped image\n C130 Airlifter')
plt.subplot(322)
plt.imshow(flip_img_v2),plt.title('Vertical flipped image\n F18 Fighter')
plt.subplot(323)
plt.imshow(flip_img_h1), plt.title('Horizontal flipped image\n C130 Airlifter')
plt.subplot(324)
plt.imshow(flip_img_h2), plt.title('Horizontal flipped image\n F18 Fighter')
plt.subplot(325)
plt.imshow(transp_img_1),plt.title('Transposed image\n C130 Airlifter')
plt.subplot(326)
plt.imshow(transp_img_2),plt.title('Transposed image\n F18 Fighter')
#################################################################
plt.show()