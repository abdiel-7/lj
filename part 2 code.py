import gudhi
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from itertools import combinations
import numpy as np
import pandas as pd
import os
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection, PolyCollection


def generate_two_disjoint_concentric_noisy_circle(n=200, radius1=1,radius2=2, center=(0, 0), noise_std=0.1):
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

def generate_two_disjoint_noisy_circle(n=200,radius=1, center1=(0,0), center2=(4,0), noise_std=0.1):
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

def generate_two_adjacent_noisy_circle(n=200,radius=1, center1=(0,0), center2=(2,0), noise_std=0.1):
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
dataset=np.array(dataset)
for i in range(0, 300,50):
  rips_complex = gudhi.RipsComplex(points=dataset[i])
  st = rips_complex.create_simplex_tree(max_dimension=2)
  PH = st.persistence()
  gudhi.plot_persistence_barcode(PH, legend=True)
  plt.show()