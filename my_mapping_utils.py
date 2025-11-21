import xarray as xr
import numpy as np

def apply_region_mapping(map_da, region_values, reg_name='layer'):
    """
    Maps region-based values (1D array) onto lat-lon grid using a region map.

    Parameters
    ----------
    map_da : xarray.DataArray
        Region mask with shape (lon, lat, region).
    region_values : np.ndarray
        1D values for each region (size = nregion).
    reg_name : str
        Dimension name for region in map_da.

    Returns
    -------
    np.ndarray
        Mapped field on (lon, lat) as a plain numpy array.
    """
    region_values = np.asarray(region_values)
    assert map_da.sizes[reg_name] == region_values.size, \
        "region_values must match the size of map_da's region dimension"

    # Multiply and sum over region → returns (lon, lat) numpy array
    mapped = (map_da * region_values).sum(dim=reg_name).values
    return mapped


def apply_region_mapping_matrix(map_da, region_matrix, reg_name='layer'):
    """
    Maps a region-based square matrix onto lat-lon grid by first treating it as a 1 x nregions vector.

    Parameters
    ----------
    map_da : xarray.DataArray
        Region mask with shape (lon, lat, region).
    region_matrix : np.ndarray
        Square matrix of size (nregions, nregions) or 1D array of size nregions.
    reg_name : str
        Name of the region dimension in map_da.

    Returns
    -------
    np.ndarray
        Mapped field on (lon, lat) as a plain numpy array.
    """
    # Ensure region_matrix is array and 2D (1 x nregions)
    region_matrix = np.asarray(region_matrix)
    # if region_matrix.ndim == 1:
    #     region_matrix = region_matrix[np.newaxis, :]
    # convert to DataArray aligned with map_da
    # region_da = xr.DataArray(region_matrix, dims=['newdim', reg_name])
    # Map to lon-lat
    mapped = np.nansum(map_da.values @ region_matrix.T, axis=-1)
    return mapped