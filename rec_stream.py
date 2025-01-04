import cupy as cp

import rec
import proc
from chunking import gpu_batch

slice_chunk = 16
proj_chunk = 16

class ReconStream():
    """Streaming reconstruction"""

    def __init__(self, args, nproj, nz, ni, theta):
        
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
                    
        self.proj_shape = (nproj,nz,ni)
        self.proj_shape_pad = (nproj,nz,n)
        self.rec_shape = (nz,n,n)

        self.cl_rec = rec.Rec(args, nproj, slice_chunk, n, center, theta)
        self.cl_proc = proc.Proc(args, ni, centeri, center)

    @gpu_batch(slice_chunk,axis_out=1,axis_inp=1)
    def proc_sino(self, res, data, dark, flat):
        """Processing a sinogram data chunk"""

        self.cl_proc.remove_outliers(data)
        self.cl_proc.remove_outliers(dark)
        self.cl_proc.remove_outliers(flat)
        res[:] = self.cl_proc.darkflat_correction(data, dark, flat)
        self.cl_proc.remove_stripe(res)
        
    @gpu_batch(proj_chunk,axis_out=0,axis_inp=0)
    def proc_proj(self, res, data):
        """Processing a projection data chunk"""

        self.cl_proc.retrieve_phase(data)
        self.cl_proc.minus_log(data)
        res[:] = self.cl_proc.pad360(data)
        return res

    @gpu_batch(slice_chunk,axis_out=0,axis_inp=1)
    def rec_sino(self,res,data):
        """Filter and backprojection via the Fourier-based method"""

        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        self.cl_rec.fbp_filter_center(data, 0)
        self.cl_rec.rec(res, data)
