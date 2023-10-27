################################################################
#Step 1 EDA: Read and Display Images
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
img_path_1 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\A10\\00c09f406d31a0cd9402862fbd26d930_0.jpg'
img_1 = cv2.imread(img_path_1)
img_path_2 = 'D:\Coding\School\img_rec_proj\PlaineData\\train\B1\\0b7a0336a2ffe85de8f2fe370e59bb60_0.jpg'
img_2 = cv2.imread(img_path_2)

# Load images with error handling
try:
    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)

    if img_1 is None or img_2 is None:
        raise Exception("One or both images could not be loaded.")
except Exception as e:
    print(f"Error loading images: {e}")
else:
    # Display images side by side
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    plt.title('Original image\n A10 aircraft')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
    plt.title('Original image\n B1 aircraft')

    plt.show()