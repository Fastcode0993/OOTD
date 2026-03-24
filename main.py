import h5py
import matplotlib.pyplot as plt
import numpy as np

data = h5py.File('D:/Intelligence/StyleColorImages.h5', 'r')
images = data['images'][:]
products = data['products'][:]

unique_labels = np.unique(products)

plt.figure(figsize=(20, 4))
for i, label in enumerate(unique_labels):
    idx = np.where(products == label)[0][0] 
    img = images[idx]
    
    if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
    
    plt.subplot(1, 10, i + 1)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.show()