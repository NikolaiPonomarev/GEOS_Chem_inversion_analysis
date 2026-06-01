import numpy as npy
import numpy.random as rnd
import numpy.linalg as nlg
import great_circle_distance as gcd


def do_region_err_random_sample(grd_lon, grd_lat,
                                reg_err,
                                cor_length,
                                scale=1.0,
                                maxsel=100,
                                **keywords):

    nx, ny = npy.shape(reg_err)
    norm_err = scale * npy.array(reg_err)

    # flatten grid to state vector
    lon_m, lat_m = npy.meshgrid(grd_lon, grd_lat, indexing='ij')  # (nx, ny)
    lon_flat = lon_m.ravel()    # (N,)
    lat_flat = lat_m.ravel()    # (N,)
    sig_flat = norm_err.ravel() # (N,) - pointwise std deviations
    N = len(lon_flat)
    print('Shape norm_err, sig_flat:', npy.shape(norm_err), npy.shape(sig_flat))
    #build covariance matrix C (N x N)
    #     diagonal: sig_i^2
    #     off-diagonal: sig_i * sig_j * exp(-d_ij / cor_length)
    dist = gcd.get_circle_distance(lon_flat, lat_flat)  # (N, N) in km
    dist_norm = dist / cor_length
    corr = npy.exp(-dist_norm)
    C = sig_flat[:, None] * corr * sig_flat[None, :]    # (N, N)

    # Cholesky decomposition C = Z Z^T
    Z = nlg.cholesky(C)  # (N, N) lower triangular

    # Draw ensemble: for each member sample G ~ N(0,1) of size (N,) 
    #the uncertainty is already included in Z through norm_err
    
    map_lst = npy.zeros((maxsel, nx, ny), dtype=float)
    for iem in range(maxsel):
        G = rnd.standard_normal(N)
        sample = Z @ G                      # (N,) correlated scaling factors
        map_lst[iem] = sample.reshape(nx, ny)

    flux_lst = map_lst
    print('map shape:', npy.shape(map_lst))

    out_dict = {
        'lon':  grd_lon,
        'lat':  grd_lat,
        'map':  map_lst,
        'flux': flux_lst,
    }

    return out_dict