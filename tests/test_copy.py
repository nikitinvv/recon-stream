import numpy as np
import cupy as cp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor,wait
import time
def _copy(res, u, st, end):
    res[st:end] = u[st:end]
    time.sleep(1)


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


# def _copy(res, u, ind):
#     res[ind] = u[ind]


def copy(u, res, nthreads=16):
    nchunk = int(np.ceil(u.shape[0]/nthreads))
    futures = [pool.submit(_copy, res, u, k*nchunk, min((k+1)*nchunk, u.shape[0])) for k in range(nthreads)]        
    wait(futures)
    return res
a = np.zeros([1024,2048,2048])

b = np.empty_like(a)
t=time.time()
for k in range(100):
    pool = ThreadPoolExecutor(16)
print(time.time()-t)
t=time.time()
copy(a,b)
# a[:]=b[:]
print(time.time()-t)
# print(a[-1,-1,-5:])
