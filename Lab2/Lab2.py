import numpy as np
from skimage import io

data_array = np.zeros((9, 400, 600))

# Exercise a
for i in range(0, 9):
    file_path = "images/car_" + str(i) + ".npy"
    data_array[i] = np.load(file_path)

# Exercise b
sum_all = np.sum(data_array)
# print(sum_all)

# Exercise c
sum_indep = np.sum(data_array, axis=(1, 2))
print(sum_indep)

# Exercise d
print("Poza cu suma maxima este:" + str(np.argmax(sum_indep)))

# Exercise e
mean_image = np.mean(data_array, axis=0)
# io.imshow(mean_image.astype(np.uint8))
# io.show()

# Exercise f
standard_dev = data_array.std()
print(standard_dev)

# Exercise g
newimages = data_array - mean_image
newimages = newimages/standard_dev
for i in range(0, 9):
    io.imshow(newimages[i].astype(np.uint8))
    io.show()

# Exercise h
sliced_img = data_array[:, 200:300, 280:400]
io.imshow(sliced_img[1].astype(np.uint8))
io.show()