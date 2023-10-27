#############################################################################

#Step 5 EDA: Analysing illumination and lighting artefacts by examining the camera effects/exposure of an #image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.color as skic
import skimage.filters as skif
import skimage.data as skid
import skimage.util as sku
import skimage.exposure as skie

img_path_1 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\C130\\0a317b1280597e9e84d2891ee88a8bb2_1.jpg'
img_1 = cv2.imread(img_path_1)
img_path_2 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\F18\\4c4a9626fea6f02ae260c2300e87d83e_7.jpg'
img_2 = cv2.imread(img_path_2)


def show(img):
    # Display the image.
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(12, 3))

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_axis_off()
    

    # Display the histogram.
    ax2.hist(img.ravel(), lw=0, bins=256)
    ax2.set_xlim(0, img.max())
    ax2.set_yticks([])
    plt.show()

show(img_1)
# adaptive histogram equalisation
show(skie.equalize_adapthist(img_1))


show(img_2)
# adaptive histogram equalisation
show(skie.equalize_adapthist(img_2))


#class 1 image
img = skic.rgb2gray(img_1)
sobimg_nheq= skif.sobel(img)
show(sobimg_nheq)
img = skic.rgb2gray(skie.equalize_adapthist(img_1))
sobimg_heq_1 = skif.sobel(img)
show(sobimg_heq_1)
#class 2 image
img = skic.rgb2gray(img_2)
sobimg_nheq= skif.sobel(img)
show(sobimg_nheq)
img = skic.rgb2gray(skie.equalize_adapthist(img_2))
sobimg_heq_2 = skif.sobel(img)
show(sobimg_heq_2)

###################################################################################