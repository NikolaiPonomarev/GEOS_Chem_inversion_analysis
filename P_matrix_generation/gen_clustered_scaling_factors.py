from numpy import *
from pylab import *
import gen_plots as gpl
import oco_feedback as ofb
import build_flux_clusters as bfc
import numpy as npy


def get_em_data(yyyy, mm, dd,datapath='./', do_debug=False):

    sdate=r'%4.4d%2.2d%2.2d0000' % (yyyy, mm, dd)
    varname=['lon', 'lat','EmisCO2_Total', 'AREA']
    flnm=datapath+'HEMCO_diagnostics.'+sdate+'.nc'
    
    lon, lat, emis_co2, sf_area=ofb.ncf_read(flnm,varname)
    emis_co2=sum(emis_co2[:,:,:], axis=0)
    
    print(shape(emis_co2), amax(emis_co2), amin(emis_co2))
    
    print('lon:', amin(lon), amax(lon))
    print('lat:', amin(lat), amax(lat))
    # from kg/s to kg/yr
    factor=365*24*3600.0
    emis_co2=factor*emis_co2
    if (do_debug):
    
        cx=cm.bwr
        cx.set_over('r')
        cx.set_under('b')
    
        gpl.plot_map(emis_co2, rlon=lon, rlat=lat, maxlon=50, minlon=-20, maxlat=35, \
                     minlat=-35, use_pcolor=1, maxv=6,minv=-6, dv=0.5, cmap=cx, ucclr='k')
        
        # show()
    return lon, lat, emis_co2, sf_area




def construct_random_em(yyyy, mm, dd, \
                        lonmin, lonmax, latmin, latmax, \
                        scale, maxsel, number_of_clusters, obs_counts_file, use_def_date=False):
    if (use_def_date):
        def_yyyy, def_mm, def_dd=2019, 1, 1
    else:
        def_yyyy, def_mm, def_dd=yyyy, mm, dd

    
    lon, lat, emis_co2, sf_area=get_em_data(def_yyyy, def_mm, def_dd,datapath='./')
    
    # select region to use
    sel_idx=npy.where((lon>=lonmin) & (lon<=lonmax))
    sel_idy=npy.where((lat>=latmin) & (lat<=latmax))
    sel_idx=npy.squeeze(sel_idx)
    sel_idy=npy.squeeze(sel_idy)

    emis_co2=npy.transpose(emis_co2)
    sf_area=npy.transpose(sf_area)
    

    mask_reg=zeros(shape(emis_co2), float)
    
    sel_lon=lon[sel_idx]
    sel_lat=lat[sel_idy]
    for ix in sel_idx:
        for iy in sel_idy:
            mask_reg[ix, iy]=1.0
    
    
            

    
    sel_flux=emis_co2*mask_reg
    # gpl.plot_map(sel_flux, lon, lat, use_pcolor=1)
    # show()

    nsel_lon=npy.size(lon)
    nsel_lat=npy.size(lat)

    out_dict = bfc.build_flux_clusters(lon, lat, emis_co2, obs_counts_file, number_of_clusters, 
                                            nens=maxsel, flux_weight=1.0, 
                                            obs_weight=1.0, scale=1.0, debug=True
                                      )
    
    out_dict.update({'sf_area':sf_area})
    
    
    return out_dict




if (__name__=='__main__'):

    yyyy, mm, dd=2019, 1, 1
    lonmin=-20 # -17.1875
    lonmax=54.375   # 50.935
    latmin=-38.0  # -9.75
    latmax=37.5   # 19.75 
    cor_length=100
    maxsel=100
    # the uncertainty for flux estimates

    scale=1#0.5
    number_of_clusters = 100
    out_dict=construct_random_em(yyyy, mm, dd, \
                                 lonmin, lonmax, latmin, latmax, scale, maxsel, number_of_clusters, \
                                 obs_counts_file='/exports/geos.ed.ac.uk/palmer_group/nponomar/GEOS_Chem_inversion_analysis_AF/GEOS_Chem_inversion_analysis/aggregated/daily_binned_prior_optimized_obs.nc'
                                 )
    
    lon=out_dict['lon']
    lat=out_dict['lat']
    reg_map=out_dict['map']
    reg_flux=out_dict['flux']
    # w=out_dict['w']
    sf_area=out_dict['sf_area']

    nem, nx, ny=shape(reg_flux)
    pid=arange(nem)+1
    sdate=r'%4.4d%2.2d%2.2d' % (yyyy, mm, dd)

    outflnm='co2_emis_rnd_clustered.'+sdate
    print('save the ensemble to:', outflnm)
    
    npy.savez(outflnm, lon=lon, lat=lat, reg_map=reg_map, \
              reg_flux=reg_flux, area=sf_area, pid=pid)
    
    
    
    figure(1)
    sel_flux=reg_map[-1]
    gpl.plot_map(transpose(sel_flux), rlon=lon, rlat=lat, minlon=-10, maxlon=50, minlat=-30, maxlat=30, use_pcolor=1, ucclr='k')
    title('Last Ensemble Member')
    savefig('mod1_clustered.png')
    
    figure(2)
    sel_flux2=reg_map[0]
    gpl.plot_map(transpose(sel_flux2), rlon=lon, rlat=lat, minlon=-10, maxlon=50, minlat=-30, maxlat=30, use_pcolor=1,ucclr='k')
    title('First Ensemble Member')
    savefig('mod2_clustered.png')


    
    

    


