import numpy as np
import matplotlib.pyplot as plt

width, height = 100, 100


cam = np.ones((width, height, 3)) * np.array([50, 50, 0])

def sphereSDF(rs, x, y, z, rad):
  # create width by height array of (x, y, z) vectors
  loc = np.ones((width, height, 3)) * np.array([x, y, z])

  # take distance between ray and sphere and subtract radius
  return np.linalg.norm(rs-loc, axis=2) - rad
  #return np.sum((rs - loc)**2, axis=2)**0.5 - rad
  

# create ndarray of vectors (x, y, 50)
rays = np.dstack(
  np.meshgrid(np.arange(width), np.arange(height))
  + [np.ones((width, height)) * 50]
  )



for i in range(1000):
  d = sphereSDF(rays, 50, 50, 100, 25)
  print(np.min(d))
  dir = (rays - cam)
  dir /= np.sum(dir**2, axis=2, keepdims=True)
  dir *= np.dstack([d]*3)
  rays += dir

dist = np.linalg.norm(rays-cam, axis=2)
dist /= np.max(dist)
print(np.max(dist), np.min(dist))

plt.pcolormesh(dist)
plt.show()

