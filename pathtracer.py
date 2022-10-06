import numpy as np
import matplotlib.pyplot as plt

scale = 2

width, height = 100*scale, 100*scale

rng = np.random.default_rng(0)

cam = np.array([50, 50, 0])

spheres = 100 * rng.random((3, 20)) + np.array([0, 0, 50])[:, np.newaxis]
radii = rng.random(spheres.shape[1]) * 50

def sphereSDF(rs, loc, rad):

  # take distance between ray and sphere and subtract radius
  # return np.linalg.norm(rs-loc, axis=2) - rad
  return np.sum((rs - loc)**2, axis=2)**0.5 - rad

def manySpheresSDF(rs, locs, rads):
  #rs - rays: WxHx3
  #locs - sphere midpoints: 3xN
  #rads - sphere radii: N

  #broadcast rs to WxHx3xN then take distance then find min distance
  return np.min(np.sum((rs[...,np.newaxis] - locs)**2, axis=-2)**0.5 - rads, axis=-1)

def sdf(rs):
  return manySpheresSDF(rs, spheres, radii)

def trace(rays, dir):
  # old_d = np.ones(rays.shape[:1])*100000
  d = sdf(rays)
  i = 0

  # normalize dir vectors
  dir /= np.sum(dir**2, axis=-1, keepdims=True)**0.5

  # trace 10 iterations
  while i < 10:
    i+=1
    rays += dir * d[..., np.newaxis]
    d = sdf(rays)

  # recursively ignore finished rays
  mask = (d > 1e-8) & (np.max(rays, axis=-1) < 1000)
  if mask.shape[0] > 0:
    rays[mask] = trace(rays[mask], dir[mask])

  return rays



# create ndarray of vectors (x, y, 50)
eye_rays = np.dstack(
  np.meshgrid(np.arange(width), np.arange(height))
  + [np.ones((width, height)) * 50]
  ) / np.array([scale, scale, 1])

eye_dir = eye_rays - cam

trace(eye_rays, eye_dir)

d = sdf(eye_rays)

depth = np.sum((eye_rays - cam)**2, axis=2)**0.5
depth = np.clip(depth, 0, 500)

plt.imshow(depth)
plt.show()
