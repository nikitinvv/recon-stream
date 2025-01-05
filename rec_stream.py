import cupy as cp
import numpy as np

import rec
import proc
from chunking import gpu_batch



class ReconStream():
    """Streaming reconstruction"""

    def __init__(self, args, pars):
        ni = pars['n']
        nproj = pars['nproj']
        nz = pars['nz']
        nflat = pars['nflat']
        ndark = pars['ndark']
        in_dtype = pars['in_dtype']
        theta = pars['theta']
        gpu_array = pars['gpu_array']
        centeri = args.rotation_axis
        
        if centeri==-1:
            centeri = ni/2

        if (args.file_type == 'double_fov'):
            n = 2*ni
            if (centeri < ni//2): # if rotation center is on the left side of the ROI
                center = ni-centeri
            else:
                center = centeri
        else:
            n = ni
            center = centeri

        if gpu_array:
            self.cl_rec = rec.Rec(args, nproj, nz, n, center, theta)
            self.cl_proc = proc.Proc(args, ni, centeri, center)
            self.ncz = nz
            self.ncproj = nproj
            # intermediate arrays with results
            self.res = [None]*3
            self.res[0] = cp.empty([nproj,nz,ni],dtype=args.dtype)
            self.res[1] = cp.empty([nproj,nz,n],dtype=args.dtype)
            self.res[2] = cp.empty([nz,n,n],dtype=args.dtype)

        else:
            ncz = args.nsino_per_chunk
            ncproj = args.nproj_per_chunk
            self.cl_rec = rec.Rec(args, nproj, ncz, n, center, theta)
            self.cl_proc = proc.Proc(args, ni, centeri, center)
            
            # allocate gpu and pinned memory buffers
            nbytes1 = 2*(nproj*ncz*ni+nflat*ncz*ni+ndark*ncz*ni)*np.dtype(in_dtype).itemsize
            nbytes1 += 2*(nproj*ncz*ni)*np.dtype(args.dtype).itemsize
            
            nbytes2 = 2*(ncproj*nz*ni)*np.dtype(args.dtype).itemsize
            nbytes2 += 2*(ncproj*nz*n)*np.dtype(args.dtype).itemsize

            nbytes3 = 2*(nproj*ncz*n)*np.dtype(args.dtype).itemsize
            nbytes3 += 2*(ncz*n*n)*np.dtype(args.dtype).itemsize

            # if rec_proc_sino
            nbytes4 = 2*(nproj*ncz*ni+nflat*ncz*ni+ndark*ncz*ni)*np.dtype(in_dtype).itemsize
            nbytes4 += 2*(ncz*n*n)*np.dtype(args.dtype).itemsize

            self.pinned_mem = cp.cuda.alloc_pinned_memory(max(nbytes1,nbytes2,nbytes3,nbytes4))
            self.gpu_mem = cp.cuda.alloc(max(nbytes1,nbytes2,nbytes3,nbytes4))

            # create CUDA streams
            self.stream1 = cp.cuda.Stream(non_blocking=True)
            self.stream2 = cp.cuda.Stream(non_blocking=True)
            self.stream3 = cp.cuda.Stream(non_blocking=True)

            # intermediate arrays with results
            self.res = [None]*3
            self.res[0] = np.empty([nproj,nz,ni],dtype=args.dtype)
            self.res[1] = np.empty([nproj,nz,n],dtype=args.dtype)
            self.res[2] = np.empty([nz,n,n],dtype=args.dtype)

            self.ncz = ncz
            self.ncproj = ncproj


    def proc_sino(self, data, dark, flat):
        @gpu_batch(self.ncz, axis_out=1, axis_inp=1)
        def _proc_sino(self, res, data, dark, flat):
            """Processing a sinogram data chunk"""

            self.cl_proc.remove_outliers(data)
            self.cl_proc.remove_outliers(dark)
            self.cl_proc.remove_outliers(flat)
            res[:] = self.cl_proc.darkflat_correction(data, dark, flat)
            self.cl_proc.remove_stripe(res)            
        return _proc_sino(self, self.res[0], data, dark, flat)

    def proc_proj(self):
        @gpu_batch(self.ncproj,axis_out=0,axis_inp=0)
        def _proc_proj(self, res, data):
            """Processing a projection data chunk"""

            self.cl_proc.retrieve_phase(data)
            self.cl_proc.minus_log(data)
            res[:] = self.cl_proc.pad360(data)            
        return _proc_proj(self, self.res[1], self.res[0])

    def rec_sino(self):
        @gpu_batch(self.ncz, axis_out=0, axis_inp=1)
        def _rec_sino(self,res,data):
            """Filter and backprojection via the Fourier-based method"""

            data = cp.ascontiguousarray(data.swapaxes(0, 1))
            self.cl_rec.fbp_filter_center(data, 0)
            self.cl_rec.rec(res, data)
        return _rec_sino(self,self.res[2],self.res[1])
    
    def proc_rec_sino(self,data, dark, flat):
        @gpu_batch(self.ncz, axis_out=0, axis_inp=1)
        def _proc_rec_sino(self, res, data, dark, flat):
            """Processing + Filter and backprojection via the Fourier-based method"""
            self.cl_proc.remove_outliers(data)
            self.cl_proc.remove_outliers(dark)
            self.cl_proc.remove_outliers(flat)
            data = self.cl_proc.darkflat_correction(data, dark, flat)# may change data type            
            self.cl_proc.remove_stripe(data)            
            self.cl_proc.minus_log(data)
            data = self.cl_proc.pad360(data) # may change data shape
            data = cp.ascontiguousarray(data.swapaxes(0, 1))            
            self.cl_rec.fbp_filter_center(data, 0)
            self.cl_rec.rec(res, data)
            # print(cp.linalg.norm(res))
        return _proc_rec_sino(self,self.res[2],data,dark,flat)
    
    def get_res(self,step=2):
        return self.res[step]
