import gudhi
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def generate_random_points(n=20, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
 x = np.random.uniform(xlim[0],xlim[1],size=n)
 y = np.random.uniform(xlim[0],xlim[1],size=n)
 return np.column_stack([x, y])

def generate_noisy_circle(n=200, radius=1, center=(0, 0), noise_std=0.1):
 theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
 ux = np.cos(theta)
 uy = np.sin(theta)
 cx, cy = center
 base = np.column_stack([cx + radius * ux, cy + radius * uy])
 noise = np.random.rand(n,2)
 points = base + noise_std*radius*noise
 return points

P1 = generate_noisy_circle()
P2 = np.vstack((P1, generate_random_points()))
rips_complex = gudhi.RipsComplex(points=P1)
st = rips_complex.create_simplex_tree(max_dimension=2)
a = st.persistence()
gudhi.plot_persistence_barcode(a, legend=True)
plt.show()
rips_complex = gudhi.RipsComplex(points=P2)
st = rips_complex.create_simplex_tree(max_dimension=2)
a = st.persistence()
gudhi.plot_persistence_barcode(a, legend=True)
plt.show()



D = cdist(P2, P2)
neighbors = (D < 0.3).sum(axis=1)
clean = neighbors > 5
P2_clean = P2[clean]
rips_complex = gudhi.RipsComplex(points=P2_clean)
st = rips_complex.create_simplex_tree(max_dimension=2)
a = st.persistence()
gudhi.plot_persistence_barcode(a, legend=True)
plt.show()