
from global_vars import args, params
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import remove_stripe 
import retrieve_phase

def darkflat_correction(data, dark, flat):
    """Dark-flat field correction"""

    dark0 = dark.astype(args.dtype, copy=False)
    flat0 = flat.astype(args.dtype, copy=False)
    flat0 /= args.bright_ratio  # == exposure_flat/exposure_proj
    # works only for processing all angles
    if args.flat_linear == 'True' and data.shape[0] == params.nproj:
        flat0_p0 = cp.mean(flat0[:flat0.shape[0]//2], axis=0)
        flat0_p1 = cp.mean(flat0[flat0.shape[0]//2+1:], axis=0)
        v = cp.linspace(0, 1, params.nproj)[..., cp.newaxis, cp.newaxis]
        flat0 = (1-v)*flat0_p0+v*flat0_p1
    else:
        flat0 = cp.mean(flat0, axis=0)
    dark0 = cp.mean(dark0, axis=0)
    res = (data.astype(args.dtype, copy=False)-dark0) / (flat0-dark0+flat0*1e-5)
    return res

def remove_outliers(data):
    """Remove outliers"""

    if (int(args.dezinger) > 0):
        w = int(args.dezinger)
        if len(data.shape) == 3:
            fdata = ndimage.median_filter(data, [w, 1, w])
        else:
            fdata = ndimage.median_filter(data, [w, w])
        data[:] = cp.where(cp.logical_and(
            data > fdata, (data - fdata) > args.dezinger_threshold), fdata, data)
    return data

def minus_log(data):
    """Taking negative logarithm"""

    data[data <= 0] = 1
    data[:] = -cp.log(data)
    data[cp.isnan(data)] = 6.0
    data[cp.isinf(data)] = 0
    return data  # reuse input memory

def pad360(data):
    """Pad data with 0 to handle 360 degrees scan"""

    if (params.centeri < params.ni//2):
        # if rotation center is on the left side of the ROI
        data[:] = data[:, :, ::-1]
    w = max(1, int(2*(params.ni-params.center)))
    # smooth transition at the border
    v = cp.linspace(1, 0, w, endpoint=False)
    v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)
    data[:, :, -w:] *= v
    # double sinogram size with adding 0
    data = cp.pad(data, ((0, 0), (0, 0), (0, data.shape[-1])), 'constant')
    return data