from os import listdir

import numpy as np

from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from PIL import Image


#2nd part of the project where we work on dataset
data = np.arange(256)
y=[-1]
print(f"Learning...")
for file in listdir("./images/train"):
    # print(file)
    image = Image.open('images/train/'+file)
    current_data = np.asarray(image)
    current_data = current_data.reshape((1, 256))
    data = np.vstack((data, current_data))
    y.append(file.split("_")[0])

clf = svm.SVC(kernel="rbf", C=5)
clf.fit(data[1:,:], y[1:])

error=0.0
count=0
print(f"Testing new data...")
for file in listdir("./images/test"):
    # print(file)
    count +=1
    image = Image.open('images/test/'+file)
    current_data = np.asarray(image)
    current_data = current_data.reshape((1, 256))
    if file.split("_")[0] != clf.predict(current_data)[0]:
        error +=1

print(f'Total tests= {count}')
print(f'Error= {error/count*100}%')
print(f'Accuracy= {(1- error/count)*100}%')
