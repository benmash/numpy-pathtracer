import numpy as np
import matplotlib.pyplot as plt

width, height = 100, 100

rng = np.random.default_rng(0)

cam = np.array([50, 50, 0])

spheres = 100 * rng.random((3, 20)) + 25
radii = rng.random((20)) * 50

def sphereSDF(rs, loc, rad):

  # take distance between ray and sphere and subtract radius
  # return np.linalg.norm(rs-loc, axis=2) - rad
  return np.sum((rs - loc)**2, axis=2)**0.5 - rad

def manySpheresSDF(rs, locs, rads):
  #rs - rays: WxHx3
  #locs - sphere midpoints: 3xN
  #rads - sphere radii: N

  #broadcast rs to WxHx3xN then take distance then find min distance
  return np.min(np.sum((rs[...,np.newaxis] - locs)**2, axis=2)**0.5 - rads, axis = 2)

def sdf(rs):
  return manySpheresSDF(rs, spheres, radii)

def trace(rays):
  for i in range(1000):
    d = sdf(rays)
    # print(np.min(d))
    dir = (rays - cam)
    dir /= np.sum(dir**2, axis=2, keepdims=True)**0.5
    dir *= d[..., np.newaxis]
    rays += dir
  return rays



# create ndarray of vectors (x, y, 50)
eye_rays = np.dstack(
  np.meshgrid(np.arange(width), np.arange(height))
  + [np.ones((width, height)) * 50]
  )

trace(eye_rays)

d = sdf(eye_rays)
surface_filter  = d < 1e-5
# d[surface_filter] = 0
# d[~surface_filter] = 1

plt.imshow(d)
plt.show()
