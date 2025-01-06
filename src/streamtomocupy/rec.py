import cupy as cp

from streamtomocupy import fourierrec
from streamtomocupy import lprec
from streamtomocupy import linerec
from streamtomocupy import fbp_filter


class Rec():
    def __init__(self, args, nproj, ncz, n, center, ngpus):

        self.cl_rec = [None]*ngpus
        self.cl_filter = [None]*ngpus
        self.wfilter = [None]*ngpus
        self.theta = [None]*ngpus
        ne = 4*n  # filter oversampling
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                if args.reconstruction_algorithm == 'fourierrec':
                    self.cl_rec[igpu] = fourierrec.FourierRec(
                        n, nproj, ncz, args.dtype)
                elif args.reconstruction_algorithm == 'lprec':
                    center += 0.5
                    self.cl_rec[igpu] = lprec.LpRec(
                        n, nproj, ncz, args.dtype)            
                elif args.reconstruction_algorithm == 'linerec':
                    self.cl_rec[igpu] = linerec.LineRec(
                        nproj, nproj, 22, ncz, n, args.dtype)
                        
                self.cl_filter[igpu] = fbp_filter.FBPFilter(
                    ne, nproj, ncz, args.dtype)
                # calculate the FBP filter with quadrature rules
        self.wfilter[0] = self.cl_filter[0].calc_filter(args.fbp_filter)
        for igpu in range(ngpus):
            with cp.cuda.Device(igpu):
                self.wfilter[igpu] = cp.asarray(self.wfilter[0])

        self.ne = ne
        self.n = n
        self.center = center

    def rec(self, res, data, theta):
        igpu = data.device.id
        stream = cp.cuda.get_current_stream()
        self.cl_rec[igpu].backprojection(res, data, theta, stream)

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""
        igpu = data.device.id
        stream = cp.cuda.get_current_stream()
        if sht == 0:
            sht = cp.tile(cp.float32(0), [data.shape[0], 1])
        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n//2, self.ne//2-self.n//2)), mode='edge')
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        # center fix
        w = self.wfilter[igpu]*cp.exp(-2*cp.pi*1j*t *
                                      (-self.center + sht[:, cp.newaxis]+self.n/2))

        self.cl_filter[igpu].filter(tmp, w, stream)
        data[:] = tmp[:, :, self.ne//2-self.n//2:self.ne//2+self.n//2]

        return data  # reuse input memory
