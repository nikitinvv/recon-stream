import cupy as cp

import fourierrec
import fbp_filter


class Rec():
    def __init__(self, args, nproj, ncz, n, center, theta):

        ne = 4*n
        theta = cp.array(theta)

        # tomography
        self.cl_rec = fourierrec.FourierRec(n, nproj, ncz, theta, args.dtype)
                
        self.cl_filter = fbp_filter.FBPFilter(
            ne, nproj, ncz, args.dtype)

        # calculate the FBP filter with quadrature rules
        self.wfilter = self.cl_filter.calc_filter(args.fbp_filter)
        
        self.ne = ne
        self.n = n
        self.center = center

    def rec(self,res,data):
        self.cl_rec.backprojection(res,data,cp.cuda.get_current_stream())

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""
        if sht==0:
            sht=cp.tile(cp.float32(0), [data.shape[0], 1])
        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n//2, self.ne//2-self.n//2)), mode='edge')
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-self.center +sht[:, cp.newaxis]+self.n/2))  # center fix

        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, self.ne//2-self.n//2:self.ne//2+self.n//2]

        return data  # reuse input memory

    
