
from global_vars import args, params
from rec_stream import ReconStream
import cupy as cp
import numpy as np

import config
import dxchange

# init parameters with default values. can be done ones
# config.write_args('test.conf')
# read parameters
config.read_args('test.conf')

proj, flat, dark, theta = dxchange.read_aps_32id('/home/beams/TOMO/conda/tomocupy/tests/data/test_data.h5')

[nproj,nz,n] = proj.shape

cl_recstream = ReconStream(nproj,nz,n,theta)
res1 = np.empty_like(proj,dtype=args.dtype)
cl_recstream.proc_sino(res1,proj,dark,flat)

res2 = np.empty_like(proj,dtype=args.dtype)
cl_recstream.proc_proj(res2,res1)

res3 = np.empty([res2.shape[1],res2.shape[2],res2.shape[2]],dtype=args.dtype)
cl_recstream.rec_sino(res3,res2)