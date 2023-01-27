import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def interpolate_3d(f, w, num_points1, num_points2, num_points3,
                   indexx, indexy, indexz, dx1, dx2, dx3):
  dim1 = indexx.shape[0]
  dim2 = indexx.shape[1]
  dim3 = indexx.shape[2]
  f_out = np.zeros((dim1, dim2, dim3))


  X = w[:,:,:,0]
  Y = w[:,:,:,1]
  Z = w[:,:,:,2]

  nsqr = num_points2 * num_points3
  for k in prange(dim3):
    for j in range(dim2):
      for i in range(dim1):
        stepsx = X[i,j,k] / dx1
        stepsy = Y[i,j,k] / dx2
        stepsz = Z[i,j,k] / dx3

        px = np.floor(stepsx)
        py = np.floor(stepsy)
        pz = np.floor(stepsz)

        ax = stepsx - px
        ay = stepsy - py
        az = stepsz - pz

        pxindex = int(indexx[i,j,k] + px)
        pyindex = int(indexy[i,j,k] + py)
        pzindex = int(indexz[i,j,k] + pz)
        pxindex_x = int(pxindex + 1)
        pyindex_y = int(pyindex + 1)
        pzindex_z = int(pzindex + 1)
  
        if (pxindex<0):
          pxindex = 0
        if (pyindex<0):
          pyindex = 0
        if (pxindex_x<0):
          pxindex_x = 0
        if (pyindex_y<0):
          pyindex_y = 0
        if (pzindex<0):
          pzindex = 0
        if (pzindex_z<0):
          pzindex_z = 0

        if (pxindex>num_points1-1):
            pxindex = num_points1-1
        if (pyindex>num_points2-1):
            pyindex = num_points2-1
        if (pxindex_x>num_points1-1):
            pxindex_x = num_points1-1
        if (pyindex_y>num_points2-1):
            pyindex_y = num_points2 -1
        if (pzindex>num_points3-1):
            pzindex = num_points3-1
        if (pzindex_z>num_points3-1):
            pzindex_z = num_points3-1

        pindex = int(pxindex * nsqr + (num_points3)*(pyindex) + pzindex)
        pindex_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex)
        pindex_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_z = int(pxindex * nsqr + (num_points2) * (pyindex) + pzindex_z)
        pindex_z_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex_z)
        pindex_z_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex_z)
        pindex_z_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex_z)

        f_out[i,j,k] = f[pindex]*(1-ax)*(1-ay)*(1-az) + \
                          f[pindex_x]*(ax)*(1-ay)*(1-az) + \
                          f[pindex_y]*(1-ax)*(ay)*(1-az) + \
                          f[pindex_xy]*(ax)*(ay)*(1-az) + \
                          f[pindex_z]*(1-ax)*(1-ay)*(az) + \
                          f[pindex_z_x]*(ax)*(1-ay)*(az) + \
                          f[pindex_z_y]*(1-ax)*(ay)*(az) + \
                          f[pindex_z_xy]*(ax)*(ay)*(az)
  return f_out


@jit(nopython=True, parallel=True)
def interpolate_3d_dual(f, w, num_points1, num_points2, num_points3,
                   indexx, indexy, indexz, dx1, dx2, dx3):
  dim1 = indexx.shape[0]
  dim2 = indexx.shape[1]
  dim3 = indexx.shape[2]
  num_nodes = dim1*dim2*dim3
  f_out = np.zeros(num_nodes)


  X = w[:,:,:,0]
  Y = w[:,:,:,1]
  Z = w[:,:,:,2]

  nsqr = num_points2 * num_points3

  for k in prange(dim3):
    for j in range(dim2):
      for i in range(dim1):
        stepsx = X[i,j,k] / dx1
        stepsy = Y[i,j,k] / dx2
        stepsz = Z[i,j,k] / dx3

        px = np.floor(stepsx)
        py = np.floor(stepsy)
        pz = np.floor(stepsz)

        ax = stepsx - px
        ay = stepsy - py
        az = stepsz - pz

        pxindex = int(indexx[i,j,k] + px)
        pyindex = int(indexy[i,j,k] + py)
        pzindex = int(indexz[i,j,k] + pz)
        pxindex_x = int(pxindex + 1)
        pyindex_y = int(pyindex + 1)
        pzindex_z = int(pzindex + 1)

        if (pxindex<0):
          pxindex = 0
        if (pyindex<0):
          pyindex = 0
        if (pxindex_x<0):
          pxindex_x = 0
        if (pyindex_y<0):
          pyindex_y = 0
        if (pzindex<0):
          pzindex = 0
        if (pzindex_z<0):
          pzindex_z = 0

        if (pxindex>num_points1-1):
            pxindex = num_points1-1
        if (pyindex>num_points2-1):
            pyindex = num_points2-1
        if (pxindex_x>num_points1-1):
            pxindex_x = num_points1-1
        if (pyindex_y>num_points2-1):
            pyindex_y = num_points2 -1
        if (pzindex>num_points3-1):
            pzindex = num_points3-1
        if (pzindex_z>num_points3-1):
            pzindex_z = num_points3-1

        pindex = int(pxindex * nsqr + (num_points3)*(pyindex) + pzindex)
        pindex_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex)
        pindex_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_z = int(pxindex * nsqr + (num_points2) * (pyindex) + pzindex_z)
        pindex_z_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex_z)
        pindex_z_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex_z)
        pindex_z_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex_z)


        f_out[pindex] += f[i,j,k]*(1-ax)*(1-ay)*(1-az)
        f_out[pindex_x] += f[i,j,k]*(ax)*(1-ay)*(1-az)
        f_out[pindex_y]  += f[i,j,k]*(1-ax)*(ay)*(1-az)
        f_out[pindex_xy] += f[i,j,k]*(ax)*(ay)*(1-az)
        f_out[pindex_z] += f[i,j,k]*(1-ax)*(1-ay)*(az)
        f_out[pindex_z_x] += f[i,j,k]*(ax)*(1-ay)*(az)
        f_out[pindex_z_y] += f[i,j,k]*(1-ax)*(ay)*(az)
        f_out[pindex_z_xy] += f[i,j,k]*(ax)*(ay)*(az)
  return f_out

@jit(nopython=True, parallel=True)
def interpolate_3d_gradient(f, w, num_points1, num_points2, num_points3,
                            indexx, indexy, indexz, dx1, dx2, dx3):
  dim1 = indexx.shape[0]
  dim2 = indexx.shape[1]
  dim3 = indexx.shape[2]
  f_out = np.zeros((dim1, dim2, dim3, 3))


  X = w[:,:,:,0]
  Y = w[:,:,:,1]
  Z = w[:,:,:,2]

  nsqr = num_points2 * num_points3
  
  for k in prange(dim3):
    for j in range(dim2):
      for i in range(dim1):
        stepsx = X[i,j,k] / dx1
        stepsy = Y[i,j,k] / dx2
        stepsz = Z[i,j,k] / dx3

        px = np.floor(stepsx)
        py = np.floor(stepsy)
        pz = np.floor(stepsz)

        ax = stepsx - px
        ay = stepsy - py
        az = stepsz - pz


        pxindex = int(indexx[i, j, k] + px)
        pyindex = int(indexy[i, j, k] + py)
        pzindex = int(indexz[i, j, k] + pz)
        pxindex_x = int(pxindex + 1)
        pyindex_y = int(pyindex + 1)
        pzindex_z = int(pzindex + 1)

        if (pxindex < 0):
          pxindex = 0
        if (pyindex < 0):
          pyindex = 0
        if (pxindex_x < 0):
          pxindex_x = 0
        if (pyindex_y < 0):
          pyindex_y = 0
        if (pzindex < 0):
          pzindex = 0
        if (pzindex_z < 0):
          pzindex_z = 0

        if (pxindex > num_points1 - 1):
          pxindex = num_points1 - 1
        if (pyindex > num_points2 - 1):
          pyindex = num_points2 - 1
        if (pxindex_x > num_points1 - 1):
          pxindex_x = num_points1 - 1
        if (pyindex_y > num_points2 - 1):
          pyindex_y = num_points2 - 1
        if (pzindex > num_points3 - 1):
          pzindex = num_points3 - 1
        if (pzindex_z > num_points3 - 1):
          pzindex_z = num_points3 - 1

        pindex = int(pxindex * nsqr + (num_points3) * (pyindex) + pzindex)
        pindex_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex)
        pindex_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_z = int(pxindex * nsqr + (num_points2) * (pyindex) + pzindex_z)
        pindex_z_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex_z)
        pindex_z_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex_z)
        pindex_z_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex_z)

        f_out[i,j,k,0] = -f[pindex]*(1-ay)*(1-az) + \
                          f[pindex_x]*(1-ay)*(1-az) - \
                          f[pindex_y]*(ay)*(1-az) + \
                          f[pindex_xy]**(ay)*(1-az) - \
                          f[pindex_z]*(1-ay)*(az) + \
                          f[pindex_z_x]*(1-ay)*(az) - \
                          f[pindex_z_y]*(ay)*(az) + \
                          f[pindex_z_xy]*(ay)*(az)

        f_out[i,j,k,1] = -f[pindex]*(1-ax)*(1-az) - \
                          f[pindex_x]*(ax)*(1-az) + \
                          f[pindex_y]*(1-ax)*(1-az) + \
                          f[pindex_xy]*(ax)*(1-az) - \
                          f[pindex_z]*(1-ax)*(az) - \
                          f[pindex_z_x]*(ax)*(az) + \
                          f[pindex_z_y]*(1-ax)*(az) + \
                          f[pindex_z_xy]*(ax)*(az)

        f_out[i,j,k] = -f[pindex]*(1-ax)*(1-ay) - \
                          f[pindex_x]*(ax)*(1-ay) - \
                          f[pindex_y]*(1-ax)*(ay) - \
                          f[pindex_xy]*(ax)*(ay) + \
                          f[pindex_z]*(1-ax)*(1-ay) + \
                          f[pindex_z_x]*(ax)*(1-ay) + \
                          f[pindex_z_y]*(1-ax)*(ay) + \
                          f[pindex_z_xy]*(ax)*(ay)
  return f_out

@jit(nopython=True, parallel=True)
def interp_dual_and_grad(f, f2, w, num_points1, num_points2, num_points3,
                         indexx, indexy, indexz, dx1, dx2, dx3, dt):
  dim1 = indexx.shape[0]
  dim2 = indexx.shape[1]
  dim3 = indexx.shape[2]
  num_nodes = dim1*dim2*dim3
  #f_out_temp = np.zeros((dim1, dim2, dim3))
  f_out = np.zeros(num_nodes)
  f2_out = np.zeros((num_nodes, 3))


  X = w[:,:,:,0]
  Y = w[:,:,:,1]
  Z = w[:,:,:,2]

  nsqr = num_points2 * num_points3
  for k in prange(dim3):
    for j in range(dim2):
      for i in range(dim1):
        stepsx = -dt*X[i,j,k] / dx1
        stepsy = -dt*Y[i,j,k] / dx2
        stepsz = -dt*Z[i,j,k] / dx3

        px = np.floor(stepsx)
        py = np.floor(stepsy)
        pz = np.floor(stepsz)

        ax = stepsx - px
        ay = stepsy - py
        az = stepsz - pz

        pxindex = int(indexx[i,j,k] + px)
        pyindex = int(indexy[i,j,k] + py)
        pzindex = int(indexz[i,j,k] + pz)
        pxindex_x = int(pxindex + 1)
        pyindex_y = int(pyindex + 1)
        pzindex_z = int(pzindex + 1)

        if (pxindex<0):
          pxindex = 0
        if (pyindex<0):
          pyindex = 0
        if (pxindex_x<0):
          pxindex_x = 0
        if (pyindex_y<0):
          pyindex_y = 0
        if (pzindex<0):
          pzindex = 0
        if (pzindex_z<0):
          pzindex_z = 0

        if (pxindex>num_points1-1):
            pxindex = num_points1-1
        if (pyindex>num_points2-1):
            pyindex = num_points2-1
        if (pxindex_x>num_points1-1):
            pxindex_x = num_points1-1
        if (pyindex_y>num_points2-1):
            pyindex_y = num_points2 -1
        if (pzindex>num_points3-1):
            pzindex = num_points3-1
        if (pzindex_z>num_points3-1):
            pzindex_z = num_points3-1

        pindex = int(pxindex * nsqr + (num_points3)*(pyindex) + pzindex)
        pindex_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex)
        pindex_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex)
        pindex_z = int(pxindex * nsqr + (num_points2) * (pyindex) + pzindex_z)
        pindex_z_x = int(pxindex_x * nsqr + (num_points3) * (pyindex) + pzindex_z)
        pindex_z_y = int(pxindex * nsqr + (num_points3) * (pyindex_y) + pzindex_z)
        pindex_z_xy = int(pxindex_x * nsqr + (num_points3) * (pyindex_y) + pzindex_z)


        f_out[pindex] += f[i,j,k]*(1-ax)*(1-ay)*(1-az)
        f_out[pindex_x] += f[i,j,k]*(ax)*(1-ay)*(1-az)
        f_out[pindex_y]  += f[i,j,k]*(1-ax)*(ay)*(1-az)
        f_out[pindex_xy] += f[i,j,k]*(ax)*(ay)*(1-az)
        f_out[pindex_z] += f[i,j,k]*(1-ax)*(1-ay)*(az)
        f_out[pindex_z_x] += f[i,j,k]*(ax)*(1-ay)*(az)
        f_out[pindex_z_y] += f[i,j,k]*(1-ax)*(ay)*(az)
        f_out[pindex_z_xy] += f[i,j,k]*(ax)*(ay)*(az)

        f2_out_index = int(nsqr*indexx[i, j, k] + (num_points3) * indexy[i, j, k] + indexz[i, j, k])
        if ((i != 0) and (i != dim1-1) and (j != 0) and (j != dim2-1) and (k != 0) and (k != dim3-1)):
          f2_out[f2_out_index, 0] = (f2[f2_out_index + nsqr] - f2[f2_out_index - nsqr]) / (2 * dx1)
          f2_out[f2_out_index, 1] = (f2[f2_out_index + num_points3] - f2[f2_out_index - num_points1]) / (2 * dx2)
          f2_out[f2_out_index, 2] = (f2[f2_out_index + 1] - f2[f2_out_index - 1]) / (2 * dx3)
        else:
          f2_out[f2_out_index, 0] = 0
          f2_out[f2_out_index, 1] = 0
          f2_out[f2_out_index, 2] = 0
  return f_out, f2_out
