
from shutil import copy2
from PIL import Image
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter


refDict = {}
labs = []
def getColor(path):

    colors = 1
    im = Image.open(path)
    out = im.convert("P", palette=Image.ADAPTIVE, colors=colors).convert('RGB')

    for count, rgb_tuple in out.getcolors():
        rgb_norm = tuple(ti / 255 for ti in rgb_tuple)
        lab = color.rgb2lab(rgb_norm)
        filename = os.path.basename(path)
        labs.append(lab)

        refDict[filename] = {"count:" : count, "rgb:" : rgb_tuple, "lab:" : lab}



def plot_clusters_mat(labs):
    data = np.array(labs)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.view_init(azim=200)
    plt.show()

    model = DBSCAN(eps=2.5, min_samples=2)
    model.fit_predict(data)
    pred = model.fit_predict(data)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=model.labels_)
    ax.view_init(azim=200)
    plt.show()
    print("number of cluster found: {}".format(len(set(model.labels_))))
    print('cluster for each point: ', model.labels_)

def k_means(labs, count):
    data = np.array(labs)
    clt = KMeans(n_clusters=8)
    a = clt.fit(data)
    centroids = clt.cluster_centers_
    labels = clt.labels_
    print(centroids)
    i = 0
    top = [x[0] for x in Counter(labels).most_common(count)]
    for file in refDict.keys():
        label = labels[i]
        refDict[file]["group"] = label
        if label in top:
            copy2("dataset/resized/resized/{}".format(file), "dataset/class_color_dilbert/{}".format(top3.index(label)))
        i+=1



if __name__ == "__main__":
    imgs_path = "dataset/resized/resized/*"
    out_file_path = "dataset/resized/colors_only.json"
    paths = glob.glob(imgs_path)
    paths = sorted(paths)
    paths = paths
    count = 3 #referes to the number of colors you wish to select
    for path in paths:
        getColor(path)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print([x[0] for x in labs])
    ax.scatter([x[0] for x in labs], [x[1] for x in labs] , [x[2] for x in labs], cmap="viridis")
    k_means(labs, count)


