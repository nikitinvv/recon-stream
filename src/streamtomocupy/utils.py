import cupy as cp
import numpy as np
from concurrent.futures import wait

def _copy(res, u, st, end):
    res[st:end] = u[st:end]

def copy(u, res, pool):
    nthreads = pool._max_workers
    nthreads = min(nthreads,u.shape[0])
    nchunk = int(np.ceil(u.shape[0]/nthreads))    
    futures = [pool.submit(_copy, res, u, k*nchunk, min((k+1)*nchunk, u.shape[0])) for k in range(nthreads)]        
    wait(futures)
    return res


place_kernel = cp.RawKernel(r'''                            
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
        f[ind] = -log(f[ind]);
        //if (isnan(f[ind]))
            //f[ind] = 6;                    
        //if (isinf(f[ind]))
            //f[ind] = 0;                                   
    }
    ''', 'place')