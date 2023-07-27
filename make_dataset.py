from typing import no_type_check
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
import cv2
import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm

def get_average_color(x):
    b, g, r = x[:, 0], x[:, 1], x[:, 2]

    return np.array([np.mean(b), np.mean(g), np.mean(r)])

df = pd.DataFrame()
phase = 'train'
dataset = 'Kligler'
img_path = os.path.join('dataset', dataset, phase, 'input')
root_path = os.path.join('dataset', dataset, phase, 'target')
paths = os.listdir(root_path)
paths.sort()
img_paths = []
gt_paths = []
background_colors = [[], [], []]

def process_img(path):
    img_paths.append(os.path.join(img_path, path))
    gt_paths.append(os.path.join(root_path, path))

    x = cv2.imread(os.path.join(root_path, path))
    h, w, c = x.shape
    x = x.flatten().reshape(h*w, c)
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(x)
    #km = KMeans(n_clusters=2)
    #km.fit(x)

    cls = gmm.predict(x.flatten().reshape(h*w, c))
    #cls = km.predict(x.flatten().reshape(h*w, c))
    cls0_colors = x[cls == 0]
    cls1_colors = x[cls == 1]

    cls0_avg_color = get_average_color(cls0_colors)
    cls1_avg_color = get_average_color(cls1_colors)


    if np.sum(cls0_avg_color)>=np.sum(cls1_avg_color):
        background_color = cls0_avg_color
        #cls = 1 - cls
    else:
        background_color = cls1_avg_color

    gmm_out = np.array([cls0_avg_color if i == 0 else cls1_avg_color for i in cls])
    # cv2.imwrite('../dataset/Jung/'+phase+'/gmm/gmm_{:s}.jpg'.format(path), gmm_out.reshape(h, w, c))
    #cv2.imwrite('../dataset/Jung/'+phase+'/kmeans/km_{:s}.jpg'.format(path), gmm_out.reshape(h, w, c))
    #cv2.imwrite('../dataset/Jung/'+phase+'/background/background_{:s}.jpg'.format(path), np.full_like(x, background_color).reshape(h, w, c))
    #cv2.imwrite('gmm/{:s}'.format(path), cls.reshape(h, w)*255)
    for i in range(3):
        background_colors[i].append(background_color[i])
    


if __name__ == '__main__':

    for path in tqdm(paths):
        process_img(path)

    df['input'] = img_paths
    df['target'] = gt_paths
    df['B'], df['G'], df['R'] = background_colors[0], background_colors[1], background_colors[2]
    os.makedirs(os.path.join('csv', dataset), exist_ok=True)
    df.to_csv(os.path.join('csv', dataset, phase + '.csv'))

