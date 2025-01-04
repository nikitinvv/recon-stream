import cupy as cp
import numpy as np
from threading import Thread

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

def _copy(res, u, st, end):
    res[st:end] = u[st:end]

def copy(u, res, nthreads=16):
    nchunk = int(np.ceil(u.shape[0]/nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copy, args=(
            res, u, k*nchunk, min((k+1)*nchunk, u.shape[0])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()
    return res

def pinned_array(array):
    """Allocate pinned memory and associate it with numpy array"""

    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
        mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src

def gpu_batch(chunk=8,axis_out=0,axis_inp=0):

    def decorator(func):
        def inner(*args, **kwargs):
            stream1 = cp.cuda.Stream(non_blocking=False)
            stream2 = cp.cuda.Stream(non_blocking=False)
            stream3 = cp.cuda.Stream(non_blocking=False)
            cl = args[0]
            out = args[1]
            inp = args[2:]

            # if out array is on gpu then just run the function
            if isinstance(out, cp.ndarray):
                func(cl, out, *inp, **kwargs)
                return
                
            # else do processing by chunks                         
            size = out.shape[axis_out]
            nchunk = int(np.ceil(size/chunk))
            
            # allocate memory for an ouput chunk on each gpu
            out_shape0 = list(out.shape)
            out_shape0[axis_out] = chunk
            out_gpu = cp.empty([2,*out_shape0], dtype=out.dtype)
            out_pinned = pinned_array(np.empty([2,*out_shape0], dtype=out.dtype))

            # determine the number of inputs and allocate memory for each input chunk
            inp_gpu = []
            inp_pinned = []
            ninp = 0
            for k in range(0, len(inp)):
                if (isinstance(inp[k], np.ndarray) or isinstance(inp[k], cp.ndarray)) and inp[k].shape[axis_inp] == size:
                    inp_shape0 = list(inp[k].shape)
                    inp_shape0[axis_inp] = chunk
                    inp_gpu.append(cp.empty(inp_shape0, dtype=inp[k].dtype))
                    inp_pinned.append(pinned_array(np.empty(inp_shape0, dtype=inp[k].dtype)))
                    ninp += 1
                else:
                    break
            inp_gpu = [inp_gpu, inp_gpu]
            inp_pinned = [inp_pinned, inp_pinned]

            # run by chunks
            for k in range(nchunk+2):
                
                if (k > 0 and k < nchunk+1):                    
                    with stream2: # processing      
                        func(cl, out_gpu[(k-1)%2], *inp_gpu[(k-1)%2], *inp[ninp:], **kwargs)       
                    
                if (k > 1):                    
                    with stream3:  # gpu->cpu copy                          
                        out_gpu[(k-2)%2].get(out=out_pinned[(k-2) % 2])
                                    
                if (k < nchunk):
                    with stream1:  # copy to pinned memory
                        st, end = k*chunk, min(size, (k+1)*chunk)
                        for j in range(ninp):                            
                            if axis_inp==0:
                                copy(inp[j][st:end],inp_pinned[k % 2][j][:end-st])
                            elif axis_inp==1:
                                copy(inp[j][:,st:end],inp_pinned[k % 2][j][:,:end-st])
                            
                        with stream1:  # cpu->gpu copy
                            for j in range(ninp):
                                inp_gpu[k % 2][j].set(inp_pinned[k % 2][j])                            
                stream3.synchronize()
                if (k > 1):
                    st, end = (k-2)*chunk, min(size, (k-1)*chunk)                    
                    if axis_out==0:
                        copy(out_pinned[(k-2) % 2][:end-st],out[st:end])
                    if axis_out==1:
                        copy(out_pinned[(k-2) % 2][:,:end-st],out[:,st:end])                    
                stream1.synchronize()
                stream2.synchronize()                
            return
        return inner
    return decorator


# @gpu_batch(8,axis_inp=0)
# def S(res, psi, shift):
#     """Shift operator"""
#     n = psi.shape[-1]
#     p = shift.copy()
#     psi = cp.pad(psi,((0,0),(n//2,n//2),(n//2,n//2)),'symmetric')
#     x = cp.fft.fftfreq(2*n).astype('float32')
#     [x, y] = cp.meshgrid(x, x)
#     pp = cp.exp(-2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
#     for k in range(100):
#         psi = cp.fft.ifft2(pp*cp.fft.fft2(psi))
#     res[:] = psi[:,n//2:-n//2,n//2:-n//2]
    
#     return


# def pinned_array(array):
#     """Allocate pinned memory and associate it with numpy array"""

#     mem = cp.cuda.alloc_pinned_memory(array.nbytes)
#     src = np.frombuffer(
#         mem, array.dtype, array.size).reshape(array.shape)
#     src[...] = array
#     return src

# cp.random.seed(10)
# a = np.random.random([24,512,512])+1j*np.random.random([24,512,512])
# a = a.astype('complex64')
# shift = np.array(np.random.random([a.shape[0], 2]), dtype='float32')+3

# res = a*0

# ap = pinned_array(a)
# shiftp = pinned_array(shift)
# resp = pinned_array(res)


# S(resp, ap, shiftp)

# res1 = cp.array(a*0)
# a = cp.array(a)
# shift = cp.array(shift)
# S(res1, a,shift)


# print(np.linalg.norm(res1))
# print(np.linalg.norm(resp))



# @gpu_batch2
# def S2(psi, shift):
#     """Shift operator"""
#     n = psi.shape[-1]
#     p = shift.copy()#[st:end]
#     res = psi.copy()
#     # if p.shape[0]!=res.shape[0]:
#         # res = cp.tile(res,(shift.shape[0],1,1))
#     res = cp.pad(res,((0,0),(n//2,n//2),(n//2,n//2)),'symmetric')
#     x = cp.fft.fftfreq(2*n).astype('float32')
#     [x, y] = cp.meshgrid(x, x)
#     pp = cp.exp(-2*cp.pi*1j * (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
#     res = cp.fft.ifft2(pp*cp.fft.fft2(res))
#     res = res[:,n//2:-n//2,n//2:-n//2]
#     return res

# import tifffile
# cp.random.seed(10)
# a = np.random.random([35,256,256])+1j*np.random.random([35,256,256])
# a = a.astype('complex64')
# shift = np.array(np.random.random([a.shape[0], 2]), dtype='float32')+3

# b = S(a,shift)
# print(np.linalg.norm(b))

# b = S2(a,shift)
# print(np.linalg.norm(b))


# [b,b0] = S(a,shift)
# [bb,bb0] = S(cp.array(a),cp.array(shift))

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(b[19].real,cmap='gray')
# plt.colorbar()
# plt.savefig('t1.png')

# plt.figure()
# plt.imshow(bb[19].real.get(),cmap='gray')
# plt.colorbar()
# plt.savefig('t.png')

# # # print(np.linalg.norm(c))
# print(np.linalg.norm(b))
# print(cp.linalg.norm(bb))
# print(np.linalg.norm(b.real-bb.get().real))


# # print(np.linalg.norm(b-c))
