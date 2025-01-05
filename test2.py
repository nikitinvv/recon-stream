import cupy as cp
import numpy as np

place1_kernel = cp.RawKernel(r'''                            
                extern "C"                
                void __global__ place(float* f, int n0, int n1, int n2)
                {
                    int tx = blockDim.x * blockIdx.x + threadIdx.x;
                    int ty = blockDim.y * blockIdx.y + threadIdx.y;
                    int tz = blockDim.z * blockIdx.z + threadIdx.z;
                    if (tx >= n0 || ty >= n1 || tz >= n2)
                        return;
                    int ind = tz*n0*n1+ty*n0+tx;
                    if (f[ind]<=0)
                        f[ind] = 1;                    
                }
                ''', 'place')

place2_kernel = cp.RawKernel(r'''                            
                extern "C"                
                void __global__ place(float* f, int n0, int n1, int n2)
                {
                    int tx = blockDim.x * blockIdx.x + threadIdx.x;
                    int ty = blockDim.y * blockIdx.y + threadIdx.y;
                    int tz = blockDim.z * blockIdx.z + threadIdx.z;
                    if (tx >= n0 || ty >= n1 || tz >= n2)
                        return;
                    int ind = tz*n0*n1+ty*n0+tx;
                    if (isnan(f[ind]))
                        f[ind] = 6;                    
                    if (isinf(f[ind]))
                        f[ind] = 0;                    
                }
                ''', 'place')

data = (cp.arange(12).reshape(2, 3, 2)-5).astype('float32')
data[0,0,0]=cp.nan
data[1,2,0]=cp.inf

sh = data.shape
# print(data)
a = (int(cp.ceil(sh[2]/32)),int(np.ceil(sh[1]/32)),sh[0])
place1_kernel(a,(32, 32, 1), (data, sh[2], sh[1], sh[0]))

place2_kernel(a,(32, 32, 1), (data, sh[2], sh[1], sh[0]))
# print(data)

#for k in range(len(idsx)):
#    arr[idsy[k],idsx[k]]=2


#a = cp.flatnonzero(mask)#mask.ravel().nonzero()[0]
#print(a)

#cp.put(arr,ids,2)#arr[ids]= 2
