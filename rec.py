import cupy as cp

from global_vars import args, params
import fourierrec
import fbp_filter


class Rec():
    def __init__(self):

        params.ne = 4*params.n
        theta = cp.array(params.theta)

        # tomography
        self.cl_rec = fourierrec.FourierRec(
            params.n, params.nproj, params.ncz, theta, args.dtype)
                
        self.cl_filter = fbp_filter.FBPFilter(
            params.ne, params.nproj, params.ncz, args.dtype)

        # calculate the FBP filter with quadrature rules
        self.wfilter = self.cl_filter.calc_filter(args.fbp_filter)

    def rec(self,res,data):
        self.cl_rec.backprojection(res,data,cp.cuda.get_current_stream())

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (params.ne//2-params.n//2, params.ne//2-params.n//2)), mode='edge')
        t = cp.fft.rfftfreq(params.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-params.center +
                                               sht[:, cp.newaxis]+params.n/2))  # center fix

        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, params.ne//2-params.n//2:params.ne//2+params.n//2]

        return data  # reuse input memory

    
