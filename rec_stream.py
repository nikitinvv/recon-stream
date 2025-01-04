import cupy as cp
import numpy as np

from chunking import gpu_batch
from global_vars import args, params
import rec
import proc

slice_chunk = 16
proj_chunk = 16

class ReconStream():
    """Streaming rec"""

    def __init__(self, nproj, nz, n, theta):
        params.theta = theta
        params.n = n
        params.nz = nz
        params.ncz = slice_chunk
        params.center = params.n/2
        params.nproj = nproj
        
        self.cl_rec = rec.Rec()  

    @gpu_batch(slice_chunk,axis_out=1,axis_inp=1)
    def proc_sino(self, res, data, dark, flat):
        """Processing a sinogram data chunk"""

        # dark flat field correrction
        data[:] = proc.remove_outliers(data)
        dark[:] = proc.remove_outliers(dark)
        flat[:] = proc.remove_outliers(flat)
        res[:] = proc.darkflat_correction(data, dark, flat)

        # remove stripes
        if args.remove_stripe_method == 'fw':
            res[:] = proc.remove_stripe_fw(res, args.fw_sigma, args.fw_filter, args.fw_level)
        elif args.remove_stripe_method == 'ti':
            res[:] = proc.remove_stripe_ti(res, args.ti_beta, args.ti_mask)
        elif args.remove_stripe_method == 'vo-all':
            res[:] = proc.remove_all_stripe(res, args.vo_all_snr, args.vo_all_la_size, args.vo_all_sm_size, args.vo_all_dim)

    @gpu_batch(proj_chunk,axis_out=0,axis_inp=0)
    def proc_proj(self, res, data):
        """Processing a projection data chunk"""

        # retrieve phase
        if args.retrieve_phase_method == 'Gpaganin' or args.retrieve_phase_method == 'paganin':
            data[:] = proc.paganin_filter(
                data,  args.pixel_size*1e-4, args.propagation_distance/10, args.energy,
                args.retrieve_phase_alpha, args.retrieve_phase_method, args.retrieve_phase_delta_beta,
                args.retrieve_phase_W*1e-4)  # units adjusted based on the tomopy implementation

        # minus log
        if args.minus_log == 'True':
            data[:] = proc.minus_log(data)

        # padding for 360 deg recon
        if args.file_type == 'double_fov':
            res[:] = proc.pad360(data)
        else:
            res[:] = data[:]
        return res

    @gpu_batch(slice_chunk,axis_out=0,axis_inp=1)
    def rec_sino(self,res,data):
        """Filter and backprojection via the Fourier-based method"""

        # swapaxes
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        # filter
        data = self.cl_rec.fbp_filter_center(data, cp.tile(np.float32(0), [data.shape[0], 1]))
        # backproj
        self.cl_rec.rec(res, data)
