import gudhi
import random
from persim import PersistenceImager
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import combinations
import numpy as np
import pandas as pd
import os
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection, PolyCollection
from sklearn.cluster import KMeans
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler




def generate_two_disjoint_concentric_noisy_circle(n=200, radius1=1,radius2=2, center=(0, 0), noise_std=1):
  theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
  ux = np.cos(theta)
  uy = np.sin(theta)
  cx, cy = center
  base = np.column_stack([cx + radius1 * ux, cy + radius1 * uy])
  noise = np.random.rand(100,2)
  points1 = base + noise_std*radius1*noise
  theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
  ux = np.cos(theta)
  uy = np.sin(theta)
  cx, cy = center
  base = np.column_stack([cx + radius2 * ux, cy + radius2 * uy])
  noise = np.random.rand(100, 2)
  points2 = base + noise_std * radius1 * noise
  points = np.vstack([points1, points2])
  return points

def generate_two_disjoint_noisy_circle(n=200,radius=1, center1=(0,0), center2=(3,0), noise_std=1):
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
    ux = np.cos(theta)
    uy = np.sin(theta)
    cx1, cy1 = center1
    base = np.column_stack([cx1 + radius * ux, cy1 + radius * uy])
    noise = np.random.rand(100, 2)
    points1 = base + noise_std * radius * noise
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
    ux = np.cos(theta)
    uy = np.sin(theta)
    cx2, cy2 = center2
    base = np.column_stack([cx2 + radius * ux, cy2 + radius * uy])
    noise = np.random.rand(100, 2)
    points2 = base + noise_std * radius * noise
    points = np.vstack([points1, points2])
    return points

def generate_two_adjacent_noisy_circle(n=200,radius=1, center1=(0,0), center2=(2,0), noise_std=1):
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
    ux = np.cos(theta)
    uy = np.sin(theta)
    cx1, cy1 = center1
    base = np.column_stack([cx1 + radius * ux, cy1 + radius * uy])
    noise = np.random.rand(100, 2)
    points1 = base + noise_std * radius * noise
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=100)
    ux = np.cos(theta)
    uy = np.sin(theta)
    cx2, cy2 = center2
    base = np.column_stack([cx2 + radius * ux, cy2 + radius * uy])
    noise = np.random.rand(100, 2)
    points2 = base + noise_std * radius * noise
    points = np.vstack([points1, points2])
    return points







def split_dgms(persistence):
    dgm0 = []
    dgm1 = []

    for dim, (b, d) in persistence:
        if d == float("inf"):
            continue
        if dim == 0:
            dgm0.append([b, d])
        elif dim == 1:
            dgm1.append([b, d])

    return np.array(dgm0), np.array(dgm1)


def diagram_stats(dgm):
    if len(dgm) == 0:
        return np.zeros(4)

    pers = dgm[:,1] - dgm[:,0]
    return np.array([
        len(pers), pers.mean(), pers.max(), pers.sum()])


def ph(persistence):
    dgm0, dgm1 = split_dgms(persistence)
    v0 = diagram_stats(dgm0)
    v1 = diagram_stats(dgm1)
    return np.concatenate([v0, v1])

dataset = []
for i in range(0,3):
    for j in range(0,100):
        if i == 0:
            pc = generate_two_disjoint_concentric_noisy_circle()
        elif i == 1:
            pc = generate_two_disjoint_noisy_circle()
        else:
            pc = generate_two_adjacent_noisy_circle()
        dataset.append(pc)
PH = []
for i in range(0, 300):
    rips_complex = gudhi.RipsComplex(points=dataset[i])
    st = rips_complex.create_simplex_tree(max_dimension=2)
    a = st.persistence()
    gudhi.plot_persistence_barcode(a, legend=True)
    plt.show()
    PH.append(ph(a))

PH = np.array(PH)
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(PH)
print(clusters)




