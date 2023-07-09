from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from PIL import Image

# making blobs data
blob_data, blob_y = make_blobs(n_samples=1000, centers=2, random_state=40)

clf = svm.SVC(kernel="linear", C=1)
clf.fit(blob_data, blob_y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    blob_data,
    plot_method="contour",
    colors="k",
    levels=[-1,0,1],
    alpha=0.5,
    linestyles=["--","-",'--'],
    ax=ax,
)

plt.scatter(blob_data[:, 0], blob_data[:, 1], c=blob_y, marker=".", s=10)
plt.show()

#making moon data
moon_data, moon_y = make_moons(n_samples=200, noise=0.05, random_state=20)

clf = svm.SVC(kernel="rbf", C=10,degree=3)
clf.fit(moon_data, moon_y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    moon_data,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", '--'],
    ax=ax,
)

plt.scatter(moon_data[:, 0], moon_data[:, 1], c=moon_y)
plt.show()


#making circle data
circle_date, circle_y = make_circles(n_samples=200, noise=0.07, factor=0.3)


clf = svm.SVC(kernel="rbf", C=5)
clf.fit(circle_date, circle_y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    circle_date,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", '--'],
    ax=ax,
)

plt.scatter(circle_date[:, 0], circle_date[:, 1], c=circle_y)
plt.show()

#making blob data with 3 centers
blob_data, blob_y = make_blobs(n_samples=1000, centers=3, cluster_std=0.50, random_state=0)

clf = svm.SVC(kernel="poly", C=1,degree=5)
clf.fit(blob_data, blob_y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    blob_data,
    plot_method="contour",
    colors="k",
    levels=[-1,0,1],
    alpha=0.5,
    linestyles=["--","-",'--'],
    ax=ax,
)

plt.scatter(blob_data[:, 0], blob_data[:, 1], c=blob_y, marker=".", s=10)
plt.show()