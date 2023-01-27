
subroutine interpolate_3d(f, w, num_points1, num_points2, num_points3,  &
indexx, indexy, indexz, dx1, dx2, dx3, num_nodes, dim1, dim2, dim3, f_out)
  implicit none
  integer :: num_nodes
  integer :: dim1, dim2, dim3
  real(8) :: f(num_nodes)
  integer :: num_points1, num_points2, num_points3
  real(8) :: w(dim1, dim2, dim3, 3)
  integer :: indexx(dim1, dim2, dim3)
  integer :: indexy(dim1, dim2, dim3)
  integer :: indexz(dim1, dim2, dim3)
  real(8) :: f_out(dim1, dim2, dim3)
  real(8) :: dx1, dx2, dx3

!f2py real(8), intent(in), dimension(num_nodes) :: f
!f2py real(8), intent(in), dimension(dim1, dim2, dim3,3) :: w
!f2py integer, intent(in)  :: num_points1, num_points2, num_points3
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexx 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexy 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexz 
!f2py real(8), intent(in) :: dx1, dx2, dx3 
!f2py integer, intent(in) :: num_nodes
!f2py integer, intent(in) :: dim1, dim2, dim3
!f2py integer, intent(in), dimension(num_nodes,3) :: gnodes 
!f2py real(8), intent(out), dimension(dim1, dim2, dim3) :: f_out

  real(8) :: X(dim1, dim2, dim3), Y(dim1, dim2, dim3), Z(dim1, dim2, dim3)
  real(8) :: stepsx, stepsy, stepsz
  real(8) :: px, py, pz
  integer :: pxindex
  integer :: pyindex
  integer :: pzindex
  integer :: pxindex_x
  integer :: pyindex_y
  integer :: pzindex_z
  integer :: nsqr
  integer :: pindex
  integer :: pindex_x
  integer :: pindex_y
  integer :: pindex_xy
  integer :: pindex_z
  integer :: pindex_z_x
  integer :: pindex_z_y
  integer :: pindex_z_xy
  real(8) :: ax
  real(8) :: ay
  real(8) :: az
  integer :: i,j,k
  integer :: iter

  !X = reshape(w(:,1), [dim1,dim2,dim3])
  !Y = reshape(w(:,2), [dim1,dim2,dim3])
  !Z = reshape(w(:,3), [dim1,dim2,dim3])
  X = w(:,:,:,1)
  Y = w(:,:,:,2)
  Z = w(:,:,:,3)

  nsqr = num_points1 * num_points2
  !$omp parallel do private(k,j,i,stepsx,stepsy,stepsz,px,py,pz,ax,ay, &
  !$omp& az,pxindex,pyindex,pzindex,pxindex_x, &
  !$omp& pyindex_y,pzindex_z,pindex,pindex_x,pindex_y,pindex_xy,pindex_z,pindex_z_x,pindex_z_y,pindex_z_xy) shared(f_out, f)
  do k = 1, dim3, 1
  do j = 1, dim2, 1
  do i = 1, dim1, 1

  stepsx = X(i,j,k) / dx1
  stepsy = Y(i,j,k) / dx2
  stepsz = Z(i,j,k) / dx3

  px = floor(stepsx)
  py = floor(stepsy)
  pz = floor(stepsz)

  ax = stepsx - px
  ay = stepsy - py
  az = stepsz - pz

  pxindex = int(indexx(i,j,k) + px) 
  pyindex = int(indexy(i,j,k) + py) 
  pzindex = int(indexz(i,j,k) + pz)
  pxindex_x = int(pxindex + 1)
  pyindex_y = int(pyindex + 1)
  pzindex_z = int(pzindex + 1)
  
  if (pxindex<0) pxindex = 0
  if (pyindex<0) pyindex = 0
  if (pxindex_x<0) pxindex_x = 0
  if (pyindex_y<0) pyindex_y = 0
  if (pzindex<0) pzindex = 0
  if (pzindex_z<0) pzindex_z = 0

  if (pxindex>num_points1-1) pxindex = num_points1-1
  if (pyindex>num_points2-1) pyindex = num_points2-1
  if (pxindex_x>num_points1-1) pxindex_x = num_points1-1
  if (pyindex_y>num_points2-1) pyindex_y = num_points2 -1
  if (pzindex>num_points3-1) pzindex = num_points3-1
  if (pzindex_z>num_points3-1) pzindex_z = num_points3-1

  pindex = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex)+1
  pindex_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex)+1

  pindex_z = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1
  pindex_z_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1

  f_out(i,j,k) = f(pindex)*(1-ax)*(1-ay)*(1-az) + &
                    f(pindex_x)*(ax)*(1-ay)*(1-az) + &
                    f(pindex_y)*(1-ax)*(ay)*(1-az) + &
                    f(pindex_xy)*(ax)*(ay)*(1-az)
  f_out(i,j,k) = f_out(i,j,k) + f(pindex_z)*(1-ax)*(1-ay)*(az) + &
                    f(pindex_z_x)*(ax)*(1-ay)*(az) + &
                    f(pindex_z_y)*(1-ax)*(ay)*(az) + &
                    f(pindex_z_xy)*(ax)*(ay)*(az)
  end do
  end do
  end do
  !$omp end parallel do

end subroutine interpolate_3d

subroutine interpolate_3d_dual(f, w, num_points1, num_points2, num_points3,  &
indexx, indexy, indexz, dx1, dx2, dx3, num_nodes, dim1, dim2, dim3, f_out)
  implicit none
  integer :: num_nodes
  integer :: dim1, dim2, dim3
  real(8) :: f(dim1,dim2,dim3)
  integer :: num_points1, num_points2, num_points3
  real(8) :: w(dim1, dim2, dim3, 3)
  integer :: indexx(dim1, dim2, dim3)
  integer :: indexy(dim1, dim2, dim3)
  integer :: indexz(dim1, dim2, dim3)
  real(8) :: f_out(num_nodes)
  real(8) :: dx1, dx2, dx3

!f2py real(8), intent(in), dimension(dim1,dim2,dim3) :: f
!f2py real(8), intent(in), dimension(dim1, dim2, dim3,3) :: w
!f2py integer, intent(in)  :: num_points1, num_points2, num_points3
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexx 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexy 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexz 
!f2py real(8), intent(in) :: num_nodes
!f2py real(8), intent(in) :: dx1, dx2, dx3 
!f2py real(8), intent(out), dimension(num_nodes) :: f_out

  real(8) :: X(dim1, dim2, dim3), Y(dim1, dim2, dim3), Z(dim1, dim2, dim3)
  real(8) :: stepsx, stepsy, stepsz
  real(8) :: px, py, pz
  integer :: pxindex
  integer :: pyindex
  integer :: pzindex
  integer :: pxindex_x
  integer :: pyindex_y
  integer :: pzindex_z
  integer :: nsqr
  integer :: pindex
  integer :: pindex_x
  integer :: pindex_y
  integer :: pindex_xy
  integer :: pindex_z
  integer :: pindex_z_x
  integer :: pindex_z_y
  integer :: pindex_z_xy
  real(8) :: ax
  real(8) :: ay
  real(8) :: az
  real(8) :: f_out_temp(dim1, dim2, dim3)
  integer :: i,j,k

  !X = reshape(w(:,1), [dim1,dim2,dim3])
  !Y = reshape(w(:,2), [dim1,dim2,dim3])
  !Z = reshape(w(:,3), [dim1,dim2,dim3])
  X = w(:,:,:,1)
  Y = w(:,:,:,2)
  Z = w(:,:,:,3)

  nsqr = num_points1 * num_points2
  
  !omp parallel do private(k,j,i)
  do k = 1, dim3, 1
  do j = 1, dim2, 1
  do i = 1, dim1, 1

  stepsx = X(i,j,k) / dx1
  stepsy = Y(i,j,k) / dx2
  stepsz = Z(i,j,k) / dx3

  px = floor(stepsx)
  py = floor(stepsy)
  pz = floor(stepsz)

  ax = stepsx - px
  ay = stepsy - py
  az = stepsz - pz

  pxindex = int(indexx(i,j,k) + px) 
  pyindex = int(indexy(i,j,k) + py) 
  pzindex = int(indexz(i,j,k) + pz)
  pxindex_x = int(pxindex + 1)
  pyindex_y = int(pyindex + 1)
  pzindex_z = int(pzindex + 1)
  
  if (pxindex<0) pxindex = 0
  if (pyindex<0) pyindex = 0
  if (pxindex_x<0) pxindex_x = 0
  if (pyindex_y<0) pyindex_y = 0
  if (pzindex<0) pzindex = 0
  if (pzindex_z<0) pzindex_z = 0

  if (pxindex>num_points1-1) pxindex = num_points1-1
  if (pyindex>num_points2-1) pyindex = num_points2-1
  if (pxindex_x>num_points1-1) pxindex_x = num_points1-1
  if (pyindex_y>num_points2-1) pyindex_y = num_points2 -1
  if (pzindex>num_points3-1) pzindex = num_points3-1
  if (pzindex_z>num_points3-1) pzindex_z = num_points3-1

  pindex = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex)+1
  pindex_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex)+1

  pindex_z = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1
  pindex_z_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1

  f_out(pindex) = f_out(pindex) + f(i,j,k)*(1-ax)*(1-ay)*(1-az)
  f_out(pindex_x) = f_out(pindex_x) + f(i,j,k)*(ax)*(1-ay)*(1-az)
  f_out(pindex_y) = f_out(pindex_y) + f(i,j,k)*(1-ax)*(ay)*(1-az)
  f_out(pindex_xy) = f_out(pindex_xy) + f(i,j,k)*(ax)*(ay)*(1-az)
  f_out(pindex_z) = f_out(pindex_z) + f(i,j,k)*(1-ax)*(1-ay)*(az)
  f_out(pindex_z_x) = f_out(pindex_z_x) + f(i,j,k)*(ax)*(1-ay)*(az)
  f_out(pindex_z_y) = f_out(pindex_z_y) + f(i,j,k)*(1-ax)*(ay)*(az)
  f_out(pindex_z_xy) = f_out(pindex_z_xy) + f(i,j,k)*(ax)*(ay)*(az)

  end do
  end do
  end do
end subroutine interpolate_3d_dual

subroutine interpolate_3d_gradient(f, w, num_points1, num_points2, num_points3,  &
indexx, indexy, indexz, dx1, dx2, dx3, num_nodes, dim1, dim2, dim3, f_out)
  implicit none
  integer :: num_nodes
  integer :: dim1, dim2, dim3
  real(8) :: f(num_nodes)
  integer :: num_points1, num_points2, num_points3
  real(8) :: w(dim1, dim2, dim3, 3)
  integer :: indexx(dim1, dim2, dim3)
  integer :: indexy(dim1, dim2, dim3)
  integer :: indexz(dim1, dim2, dim3)
  real(8) :: f_out(dim1, dim2, dim3, 3)
  real(8) :: dx1, dx2, dx3

!f2py real(8), intent(in), dimension(num_nodes) :: f
!f2py real(8), intent(in), dimension(dim1, dim2, dim3,3) :: w
!f2py integer, intent(in)  :: num_points1, num_points2, num_points3
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexx 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexy 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexz 
!f2py real(8), intent(in) :: num_nodes
!f2py real(8), intent(in) :: dx1, dx2, dx3 
!f2py real(8), intent(out), dimension(dim1, dim2, dim3, 3) :: f_out

  real(8) :: X(dim1, dim2, dim3), Y(dim1, dim2, dim3), Z(dim1, dim2, dim3)
  real(8) :: stepsx, stepsy, stepsz
  real(8) :: px, py, pz
  integer :: pxindex
  integer :: pyindex
  integer :: pzindex
  integer :: pxindex_x
  integer :: pyindex_y
  integer :: pzindex_z
  integer :: nsqr
  integer :: pindex
  integer :: pindex_x
  integer :: pindex_y
  integer :: pindex_xy
  integer :: pindex_z
  integer :: pindex_z_x
  integer :: pindex_z_y
  integer :: pindex_z_xy
  real(8) :: ax
  real(8) :: ay
  real(8) :: az
  integer :: i,j,k

  !X = reshape(w(:,1), [dim1,dim2,dim3])
  !Y = reshape(w(:,2), [dim1,dim2,dim3])
  !Z = reshape(w(:,3), [dim1,dim2,dim3])
  X = w(:,:,:,1)
  Y = w(:,:,:,2)
  Z = w(:,:,:,3)

  nsqr = num_points1 * num_points2
  
  !omp parallel do private(k,j,i)
  do k = 1, dim3, 1
  do j = 1, dim2, 1
  do i = 1, dim1, 1

  stepsx = X(i,j,k) / dx1
  stepsy = Y(i,j,k) / dx2
  stepsz = Z(i,j,k) / dx3

  px = floor(stepsx)
  py = floor(stepsy)
  pz = floor(stepsz)

  ax = stepsx - px
  ay = stepsy - py
  az = stepsz - pz

  pxindex = int(indexx(i,j,k) + px) 
  pyindex = int(indexy(i,j,k) + py) 
  pzindex = int(indexz(i,j,k) + pz)
  pxindex_x = int(pxindex + 1)
  pyindex_y = int(pyindex + 1)
  pzindex_z = int(pzindex + 1)
  
  if (pxindex<0) pxindex = 0
  if (pyindex<0) pyindex = 0
  if (pxindex_x<0) pxindex_x = 0
  if (pyindex_y<0) pyindex_y = 0
  if (pzindex<0) pzindex = 0
  if (pzindex_z<0) pzindex_z = 0

  if (pxindex>num_points1-1) pxindex = num_points1-1
  if (pyindex>num_points2-1) pyindex = num_points2-1
  if (pxindex_x>num_points1-1) pxindex_x = num_points1-1
  if (pyindex_y>num_points2-1) pyindex_y = num_points2 -1
  if (pzindex>num_points3-1) pzindex = num_points3-1
  if (pzindex_z>num_points3-1) pzindex_z = num_points3-1

  pindex = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex)+1
  pindex_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex)+1

  pindex_z = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1
  pindex_z_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1

  f_out(i,j,k,1) = f(pindex)*(-1)*(1-ay)*(1-az) + &
                  f(pindex_x)*(1)*(1-ay)*(1-az) + &
                  f(pindex_y)*(-1)*(ay)*(1-az) + &
                  f(pindex_xy)*(1)*(ay)*(1-az) + &
                  f(pindex_z)*(-1)*(1-ay)*(az) + &
                  f(pindex_z_x)*(1)*(1-ay)*(az) + &
                  f(pindex_z_y)*(-1)*(ay)*(az) + &
                  f(pindex_z_xy)*(1)*(ay)*(az)
  f_out(i,j,k,2) = f(pindex)*(1-ax)*(-1)*(1-az) + &
                  f(pindex_x)*(ax)*(-1)*(1-az) + &
                  f(pindex_y)*(1-ax)*(1)*(1-az) + &
                  f(pindex_xy)*(ax)*(1)*(1-az) + &
                  f(pindex_z)*(1-ax)*(-1)*(az) + &
                  f(pindex_z_x)*(ax)*(-1)*(az) + &
                  f(pindex_z_y)*(1-ax)*(1)*(az) + &
                  f(pindex_z_xy)*(ax)*(1)*(az)
  f_out(i,j,k,3) = f(pindex)*(1-ax)*(1-ay)*(-1) + &
                  f(pindex_x)*(ax)*(1-ay)*(-1) + &
                  f(pindex_y)*(1-ax)*(ay)*(-1) + &
                  f(pindex_xy)*(ax)*(ay)*(-1) + &
                  f(pindex_z)*(1-ax)*(1-ay)*(1) + &
                  f(pindex_z_x)*(ax)*(1-ay)*(1) + &
                  f(pindex_z_y)*(1-ax)*(ay)*(1) + &
                  f(pindex_z_xy)*(ax)*(ay)*(1)
  end do
  end do
  end do
end subroutine interpolate_3d_gradient

subroutine interp_dual_and_grad(f, f2, w, num_points1, num_points2, num_points3,  &
indexx, indexy, indexz, dx1, dx2, dx3, dt, num_nodes, dim1, dim2, dim3, f_out, f2_out)
  implicit none
  integer :: num_nodes
  integer :: dim1, dim2, dim3
  real(8) :: f(dim1,dim2,dim3)
  real(8) :: f2(num_nodes)
  integer :: num_points1, num_points2, num_points3
  real(8) :: w(dim1, dim2, dim3, 3)
  integer :: indexx(dim1, dim2, dim3)
  integer :: indexy(dim1, dim2, dim3)
  integer :: indexz(dim1, dim2, dim3)
  real(8) :: f_out(num_nodes)
  real(8) :: f2_out(num_nodes, 3)
  real(8) :: dx1, dx2, dx3, dt

!f2py real(8), intent(in), dimension(dim1,dim2,dim3) :: f
!f2py real(8), intent(in), dimension(num_nodes) :: f2
!f2py real(8), intent(in), dimension(dim1, dim2, dim3,3) :: w
!f2py integer, intent(in)  :: num_points1, num_points2, num_points3
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexx 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexy 
!f2py integer, intent(in), dimension(dim1, dim2, dim3) :: indexz 
!f2py real(8), intent(in) :: num_nodes
!f2py real(8), intent(in) :: dx1, dx2, dx3, dt 
!f2py real(8), intent(out), dimension(num_nodes) :: f_out
!f2py real(8), intent(out), dimension(num_nodes,3) :: f2_out

  real(8) :: X(dim1, dim2, dim3), Y(dim1, dim2, dim3), Z(dim1, dim2, dim3)
  real(8) :: stepsx, stepsy, stepsz
  real(8) :: px, py, pz
  integer :: pxindex
  integer :: pyindex
  integer :: pzindex
  integer :: pxindex_x
  integer :: pyindex_y
  integer :: pzindex_z
  integer :: nsqr
  integer :: pindex
  integer :: pindex_x
  integer :: pindex_y
  integer :: pindex_xy
  integer :: pindex_z
  integer :: pindex_z_x
  integer :: pindex_z_y
  integer :: pindex_z_xy
  integer :: f2_out_index 
  real(8) :: ax
  real(8) :: ay
  real(8) :: az
  real(8) :: f_out_temp(dim1, dim2, dim3)
  integer :: i,j,k

  !X = reshape(w(:,1), [dim1,dim2,dim3])
  !Y = reshape(w(:,2), [dim1,dim2,dim3])
  !Z = reshape(w(:,3), [dim1,dim2,dim3])
  X = w(:,:,:,1)
  Y = w(:,:,:,2)
  Z = w(:,:,:,3)

  nsqr = num_points1 * num_points2
  
  !$omp parallel do private(k,j,i,stepsx,stepsy,stepsz,px,py,pz,ax,ay, &
  !$omp& az,pxindex,pyindex,pzindex,pxindex_x, f2_out_index, &
  !$omp& pyindex_y,pzindex_z,pindex,pindex_x,pindex_y,pindex_xy,pindex_z, &
  !$omp& pindex_z_x,pindex_z_y,pindex_z_xy) shared(f2_out,f) reduction(+:f_out)
  do k = 1, dim3, 1
  do j = 1, dim2, 1
  do i = 1, dim1, 1

  stepsx = -1 * dt * X(i,j,k) / dx1
  stepsy = -1 * dt * Y(i,j,k) / dx2
  stepsz = -1 * dt * Z(i,j,k) / dx3

  px = floor(stepsx)
  py = floor(stepsy)
  pz = floor(stepsz)

  ax = stepsx - px
  ay = stepsy - py
  az = stepsz - pz

  pxindex = int(indexx(i,j,k) + px) 
  pyindex = int(indexy(i,j,k) + py) 
  pzindex = int(indexz(i,j,k) + pz)
  pxindex_x = int(pxindex + 1)
  pyindex_y = int(pyindex + 1)
  pzindex_z = int(pzindex + 1)
  
  if (pxindex<0) pxindex = 0
  if (pyindex<0) pyindex = 0
  if (pxindex_x<0) pxindex_x = 0
  if (pyindex_y<0) pyindex_y = 0
  if (pzindex<0) pzindex = 0
  if (pzindex_z<0) pzindex_z = 0

  if (pxindex>num_points1-1) pxindex = num_points1-1
  if (pyindex>num_points2-1) pyindex = num_points2-1
  if (pxindex_x>num_points1-1) pxindex_x = num_points1-1
  if (pyindex_y>num_points2-1) pyindex_y = num_points2 -1
  if (pzindex>num_points3-1) pzindex = num_points3-1
  if (pzindex_z>num_points3-1) pzindex_z = num_points3-1

  pindex = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex)+1
  pindex_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex)+1
  pindex_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex)+1

  pindex_z = int(pxindex + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_x = int(pxindex_x + (num_points1)*(pyindex) + nsqr*pzindex_z)+1
  pindex_z_y = int(pxindex + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1
  pindex_z_xy = int(pxindex_x + (num_points1)*(pyindex_y) + nsqr*pzindex_z)+1

  f_out(pindex) = f_out(pindex) + f(i,j,k)*(1-ax)*(1-ay)*(1-az)
  f_out(pindex_x) = f_out(pindex_x) + f(i,j,k)*(ax)*(1-ay)*(1-az)
  f_out(pindex_y) = f_out(pindex_y) + f(i,j,k)*(1-ax)*(ay)*(1-az)
  f_out(pindex_xy) = f_out(pindex_xy) + f(i,j,k)*(ax)*(ay)*(1-az)
  f_out(pindex_z) = f_out(pindex_z) + f(i,j,k)*(1-ax)*(1-ay)*(az)
  f_out(pindex_z_x) = f_out(pindex_z_x) + f(i,j,k)*(ax)*(1-ay)*(az)
  f_out(pindex_z_y) = f_out(pindex_z_y) + f(i,j,k)*(1-ax)*(ay)*(az)
  f_out(pindex_z_xy) = f_out(pindex_z_xy) + f(i,j,k)*(ax)*(ay)*(az)

  f2_out_index = int(indexx(i,j,k) + (num_points1)*indexy(i,j,k) + nsqr*indexz(i,j,k))+1
  
!   f2_out(f2_out_index,1) = f2(pindex)*(-1/dx1)*(1-ay)*(1-az) + &
!                   f2(pindex_x)*(1/dx1)*(1-ay)*(1-az) + &
!                   f2(pindex_y)*(-1/dx1)*(ay)*(1-az) + &
!                   f2(pindex_xy)*(1/dx1)*(ay)*(1-az) + &
!                   f2(pindex_z)*(-1/dx1)*(1-ay)*(az) + &
!                   f2(pindex_z_x)*(1/dx1)*(1-ay)*(az) + &
!                   f2(pindex_z_y)*(-1/dx1)*(ay)*(az) + &
!                   f2(pindex_z_xy)*(1/dx1)*(ay)*(az)
!   f2_out(f2_out_index,2) = f2(pindex)*(1-ax)*(-1/dx2)*(1-az) + &
!                   f2(pindex_x)*(ax)*(-1/dx2)*(1-az) + &
!                   f2(pindex_y)*(1-ax)*(1/dx2)*(1-az) + &
!                   f2(pindex_xy)*(ax)*(1/dx2)*(1-az) + &
!                   f2(pindex_z)*(1-ax)*(-1/dx2)*(az) + &
!                   f2(pindex_z_x)*(ax)*(-1/dx2)*(az) + &
!                   f2(pindex_z_y)*(1-ax)*(1/dx2)*(az) + &
!                   f2(pindex_z_xy)*(ax)*(1/dx2)*(az)
!   f2_out(f2_out_index,3) = f2(pindex)*(1-ax)*(1-ay)*(-1/dx3) + &
!                   f2(pindex_x)*(ax)*(1-ay)*(-1/dx3) + &
!                   f2(pindex_y)*(1-ax)*(ay)*(-1/dx3) + &
!                   f2(pindex_xy)*(ax)*(ay)*(-1/dx3) + &
!                   f2(pindex_z)*(1-ax)*(1-ay)*(1/dx3) + &
!                   f2(pindex_z_x)*(ax)*(1-ay)*(1/dx3) + &
!                   f2(pindex_z_y)*(1-ax)*(ay)*(1/dx3) + &
!                   f2(pindex_z_xy)*(ax)*(ay)*(1/dx3)
  if ((i/=1).and.(i/=dim1).and.(j/=1).and.(j/=dim2).and.(k/=1).and.(k/=dim3)) then
        f2_out(f2_out_index,1) = (f2(f2_out_index+1) - f2(f2_out_index-1))/(2*dx1)
        f2_out(f2_out_index,2) = (f2(f2_out_index+num_points1) - f2(f2_out_index-num_points1))/(2*dx2)
        f2_out(f2_out_index,3) = (f2(f2_out_index+nsqr) - f2(f2_out_index-nsqr))/(2*dx3)
  else
        f2_out(f2_out_index,1) = 0.
        f2_out(f2_out_index,2) = 0.
        f2_out(f2_out_index,3) = 0.
  end if

  end do
  end do
  end do
  !$omp end parallel do

end subroutine interp_dual_and_grad
