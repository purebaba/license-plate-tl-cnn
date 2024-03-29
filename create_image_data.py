import numpy as np
import cv2
import pickle

# create train label
a = np.loadtxt('label.txt')
b = np.zeros([5000, 65])
for i in range(5000):
    for j in range(7):
        b[i, int(a[i, j])] = int(a[i, j])

# create image train data
path = './data'
img_data = np.zeros([5000, 36, 100, 1])
for i in range(5000):
    img_path = path + '/' + str(i).zfill(4) + ".jpg"
    print(img_path)
    b = cv2.imread(img_path)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    b = np.reshape(b, (36, 100, 1))
    img_data[i, :, :, :] = b
    print('num:' + str(i))

output = open('img_data_gray.pkl', 'wb')
pickle.dump(img_data, output)
