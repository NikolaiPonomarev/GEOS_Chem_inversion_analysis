import numpy as npy
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import great_circle_distance as gcd
from scipy.ndimage import label, gaussian_filter, maximum_filter, generate_binary_structure
import heapq

def build_flux_clusters( lon, lat, flux2d, obs_counts_file, n_clusters,
    nens=10,
    flux_weight=1.0,
    obs_weight=1.0,
    scale=1.0,
    debug=False,
    # smooth_sigma=1.0
    ):
    """
    Produce clustered ensemble perturbations.
    output:
    {
        "map": (nens, lon, lat) perturbation maps
        "flux": (nens, lon, lat) perturbed fluxes
    }
    """
    print('Lon and lat shapes:', lon.shape, lat.shape)

    # nobs
    ds = Dataset(obs_counts_file, "r")
    obs_counts = ds.variables["counts"][:]  # (time, lat, lon)
    ds.close()

    obs_mean = npy.nanmean(obs_counts, axis=0)
    obs_mean = npy.nan_to_num(obs_mean, nan=0.0)
    obs_mean = gaussian_filter(obs_mean, sigma=2.0)


    # ensure obs is (lon, lat)
    if obs_mean.shape == (len(lon), len(lat)):
        pass
    elif obs_mean.shape == (len(lat), len(lon)):
        obs_mean = obs_mean.T

    if npy.std(obs_mean) > 0:
        obs_mean = (obs_mean - npy.mean(obs_mean)) / npy.std(obs_mean)
    else:
        obs_mean = obs_mean * 0.0

    #normalize the number of observations field   
    if npy.std(obs_mean) > 0:
        obs_mean = (obs_mean - npy.mean(obs_mean)) / npy.std(obs_mean)

    #flux info
    flux_norm = (flux2d - npy.mean(flux2d)) / (npy.std(flux2d))
    # print('Flux norm stats:', flux_norm.min(), flux_norm.max())
    # print("Flux norm log10 span:", npy.log10(npy.max(npy.abs(flux_norm))), npy.log10(npy.min(npy.abs(flux_norm))))
    #compute flux gradient
    gy, gx = npy.gradient(flux_norm)
    grad = npy.sqrt(gx**2 + gy**2)

    if npy.std(grad) > 0:
        grad = (grad - npy.mean(grad)) / npy.std(grad)

    #compute combined score
    score = (
        flux_weight * flux_norm +

        0.5 * flux_weight * grad + #dampen the gradient contribution to avoid noise

        obs_weight * obs_mean
    )

    # score = gaussian_filter(score, sigma=smooth_sigma)

    #clustering
    score = gaussian_filter(score, sigma=2.0)
    # cluster_map = region_growing_clustering(score, lon, lat, n_clusters=n_clusters)
    # cluster_map = watershed_clustering(score, lon, lat, n_clusters=n_clusters)
    cluster_map = contour_clustering(score, lon, lat, n_clusters=n_clusters)
    #sample perturbations for each cluster
    ny, nx = cluster_map.shape
    n_clusters = npy.max(cluster_map) + 1
    print("Number of created clusters:", n_clusters)
    
    cluster_vals = npy.random.normal(0.0, scale, size=(n_clusters, nens)) # mean is 0, std is scale
    #(nens, n_clusters)
    cluster_vals = cluster_vals.T

    # directly map cluster IDs onto grid
    map_lst = cluster_vals[:, cluster_map]
    # perturb fluxes
    flux_lst = flux2d[npy.newaxis, :, :] * (1.0 + map_lst)

    if debug:
        plot_cluster_debug(lon, lat, flux2d, flux_norm, grad, obs_mean, score, cluster_map)

    return {
        "map": map_lst,
        "flux": flux_lst,
        "lon": lon,
        "lat": lat
    }




def region_growing_clustering(score, lon, lat, n_clusters=1000, max_cluster_fraction=0.002):
    nlon, nlat = score.shape  # explicit, no ambiguity
    total_cells = nlon * nlat
    max_cluster_size = max(1, int(total_cells * max_cluster_fraction))

    # flat index helpers — first axis is lon, second is lat
    def idx2lonlat(idx):
        return divmod(idx, nlat)   # → (ilon, ilat)

    def lonlat2idx(ilon, ilat):
        return ilon * nlat + ilat

    cluster_map = -npy.ones((nlon, nlat), dtype=int)
    score_flat = score.ravel()
    unassigned = npy.ones(total_cells, dtype=bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    threshold_score = 0.5 * npy.std(score_flat)

    current_cluster = 0
    seed_order = npy.argsort(-score_flat)

    for seed_idx in seed_order:
        if current_cluster >= n_clusters:
            break

        if not unassigned[seed_idx]:
            continue

        cluster_map.ravel()[seed_idx] = current_cluster
        unassigned[seed_idx] = False
        region_size = 1
        front = [seed_idx]

        while front:
            new_front = []
            for idx in front:
                if region_size >= max_cluster_size:
                    break

                ilon, ilat = idx2lonlat(idx)

                for dlon, dlat in neighbors:
                    nilon, nilat = ilon + dlon, ilat + dlat

                    if nilon < 0 or nilon >= nlon or nilat < 0 or nilat >= nlat:
                        continue

                    nidx = lonlat2idx(nilon, nilat)

                    if not unassigned[nidx]:
                        continue
                    if region_size >= max_cluster_size:
                        break
                    if score_flat[nidx] < threshold_score:
                        continue

                    cluster_map.ravel()[nidx] = current_cluster
                    unassigned[nidx] = False
                    new_front.append(nidx)
                    region_size += 1

            front = new_front

        current_cluster += 1

    # --- fallback: assign unassigned cells to nearest cluster ---
    cluster_map_flat = cluster_map.ravel()
    unassigned_idx = npy.where(unassigned)[0]
    print(f"Number of unassigned cells after region growing: {len(unassigned_idx)}")

    # precompute cluster centroids for fast fallback (avoids full npy.where per cluster)
    cluster_lons = npy.zeros(current_cluster)
    cluster_lats = npy.zeros(current_cluster)
    for c in range(current_cluster):
        mask = cluster_map_flat == c
        ilons, ilats = npy.divmod(npy.where(mask)[0], nlat)
        cluster_lons[c] = npy.mean(lon[ilons])
        cluster_lats[c] = npy.mean(lat[ilats])

    for idx in unassigned_idx:
        ilon, ilat = idx2lonlat(idx)
        dists = gcd.get_circle_distance_xy(
            lon[ilon], lat[ilat],       # scalar point
            cluster_lons, cluster_lats  # all centroids at once
        )
        cluster_map_flat[idx] = npy.argmin(dists)

    return cluster_map



def watershed_clustering(score, lon, lat, n_clusters=1000):
    """
    Pure scipy/numpy watershed segmentation.
    Same input/output contract as region_growing_clustering:
    score: (nlon, nlat), returns cluster_map: (nlon, nlat) with 0-based integer cluster IDs.
    """
    nlon, nlat = score.shape

    # --- find local maxima as seeds ---
    # min_distance between peaks derived from desired cluster count
    min_dist = max(2, int(npy.sqrt(nlon * nlat / (n_clusters * npy.pi))))
    print(f"Watershed min_distance between seeds: {min_dist} cells")

    # a cell is a local max if it equals the max in its neighbourhood
    footprint_size = 2 * min_dist + 1
    local_max_val = maximum_filter(score, size=footprint_size, mode='nearest')
    is_local_max = (score == local_max_val)

    # get seed coordinates, sorted by score descending, take top n_clusters
    seed_lons, seed_lats = npy.where(is_local_max)
    seed_scores = score[seed_lons, seed_lats]
    order = npy.argsort(-seed_scores)
    seed_lons = seed_lons[order[:n_clusters]]
    seed_lats = seed_lats[order[:n_clusters]]
    n_seeds = len(seed_lons)
    print(f"Number of seeds found: {n_seeds}")

    # --- priority-queue flood fill (watershed) ---
    cluster_map = -npy.ones((nlon, nlat), dtype=int)
    visited = npy.zeros((nlon, nlat), dtype=bool)

    # heap entries: (-score, ilon, ilat, cluster_id)
    # negative score because heapq is a min-heap — we want highest score first
    heap = []
    for c, (ilon, ilat) in enumerate(zip(seed_lons, seed_lats)):
        cluster_map[ilon, ilat] = c
        visited[ilon, ilat] = True
        heapq.heappush(heap, (-score[ilon, ilat], ilon, ilat, c))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1,-1), (-1, 1), (1,-1), (1, 1)]  # 8-connectivity for smoother boundaries

    while heap:
        neg_s, ilon, ilat, c = heapq.heappop(heap)

        for dlon, dlat in neighbors:
            nilon, nilat = ilon + dlon, ilat + dlat

            if nilon < 0 or nilon >= nlon or nilat < 0 or nilat >= nlat:
                continue
            if visited[nilon, nilat]:
                continue

            visited[nilon, nilat] = True
            cluster_map[nilon, nilat] = c
            heapq.heappush(heap, (-score[nilon, nilat], nilon, nilat, c))

    n_found = int(npy.max(cluster_map)) + 1
    print(f"Number of created clusters: {n_found}")

    return cluster_map




def contour_clustering(score, lon, lat, n_clusters=100, min_cluster_fraction=0.001):
    nlon, nlat = score.shape
    total_cells = nlon * nlat
    min_size = max(1, int(total_cells * min_cluster_fraction))

    score_min, score_max = score.min(), score.max()
    lo, hi = score_min, score_max
    best_map = None
    best_n = 0
    best_threshold = (lo + hi) / 2.0

    for _ in range(40):
        threshold = (lo + hi) / 2.0

        binary = (score >= threshold).astype(int)
        labelled, n_found = label(binary)

        # drop regions smaller than min_size
        sizes = npy.bincount(labelled.ravel())
        for sl in range(1, len(sizes)):
            if sizes[sl] < min_size:
                labelled[labelled == sl] = 0

        valid_labels = npy.unique(labelled)
        valid_labels = valid_labels[valid_labels > 0]
        n_valid = len(valid_labels)

        # FIXED: lower threshold → more regions, raise threshold → fewer
        if n_valid < n_clusters:
            lo = threshold  # too few → lower threshold
        else:
            hi = threshold  # too many → raise threshold

        if best_map is None or abs(n_valid - n_clusters) < abs(best_n - n_clusters):
            best_n = n_valid
            best_map = labelled.copy()
            best_threshold = threshold

        print(f"  threshold={threshold:.4f}, n_valid={n_valid}")

    print(f"Best threshold: {best_threshold:.4f}, clusters found: {best_n}")

    # remap to contiguous 0-based IDs; -1 = unassigned
    cluster_map = -npy.ones((nlon, nlat), dtype=int)
    flat_best = best_map.ravel()
    unique_ids = npy.unique(flat_best)
    unique_ids = unique_ids[unique_ids > 0]
    for new_id, old_id in enumerate(unique_ids):
        cluster_map[best_map == old_id] = new_id
    n_clusters_found = len(unique_ids)

    # flood-fill unassigned cells from all assigned boundaries
    visited = cluster_map >= 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1,-1), (-1, 1), (1,-1), (1, 1)]

    heap = []
    # seed from every assigned cell that has an unassigned neighbour
    assigned_lons, assigned_lats = npy.where(visited)
    for ilon, ilat in zip(assigned_lons, assigned_lats):
        for dlon, dlat in neighbors:
            nilon, nilat = ilon + dlon, ilat + dlat
            if 0 <= nilon < nlon and 0 <= nilat < nlat and not visited[nilon, nilat]:
                heapq.heappush(heap, (-score[ilon, ilat], ilon, ilat,
                                      cluster_map[ilon, ilat]))
                break  # only need to push each assigned boundary cell once

    while heap:
        neg_s, ilon, ilat, c = heapq.heappop(heap)
        for dlon, dlat in neighbors:
            nilon, nilat = ilon + dlon, ilat + dlat
            if nilon < 0 or nilon >= nlon or nilat < 0 or nilat >= nlat:
                continue
            if visited[nilon, nilat]:
                continue
            visited[nilon, nilat] = True
            cluster_map[nilon, nilat] = c
            heapq.heappush(heap, (-score[nilon, nilat], nilon, nilat, c))

    print(f"Number of created clusters: {n_clusters_found}")
    return cluster_map
#debug plotting

def plot_map_ax( ax, lon, lat, data, title='', cmap='viridis', use_symlog=False, linthresh=1e-12):

    lon2d, lat2d = npy.meshgrid(lon, lat)

    if use_symlog:

        vmax = npy.nanmax(npy.abs(data))

        norm = SymLogNorm(
            linthresh=linthresh,
            vmin=-vmax,
            vmax=vmax
        )

        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            data.T,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree()
        )

    else:

        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            data.T,
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )

    ax.coastlines(linewidth=0.7)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    ax.set_title(title)

    plt.colorbar( pcm, ax=ax, orientation='vertical', shrink=0.8)


def plot_cluster_debug( lon, lat, flux2d, flux_norm, grad, obs_mean, score,cluster_map):

    fig, axs = plt.subplots( 2, 3, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    axs = axs.flatten()
    print('Min/max flux:', npy.nanmin(flux2d), npy.nanmax(flux2d))
    plot_map_ax( axs[0], lon, lat,flux2d,
        title='Flux',
        cmap='RdBu_r',
        use_symlog=True,
        linthresh=1e-12
    )
    print('Min/max flux norm:', npy.nanmin(flux_norm), npy.nanmax(flux_norm))
    plot_map_ax(axs[1],lon,lat, flux_norm,
        title='Normalized Flux',
        cmap='RdBu_r',
        use_symlog=True,
        linthresh=1e-5
    )
    print('Min/max flux gradient:', npy.nanmin(grad), npy.nanmax(grad))
    plot_map_ax( axs[2], lon, lat, grad,
        title='Normalized Flux Gradient',
        cmap='RdBu_r',
        use_symlog=True,
        linthresh=1e-5
    )
    print('Min/max observation density:', npy.nanmin(obs_mean), npy.nanmax(obs_mean))
    plot_map_ax(axs[3], lon, lat, obs_mean,
        title='Observation Density (Gaussian smoothed and normalized)',
        cmap='viridis',
        use_symlog=False
    )
    print('Min/max score:', npy.nanmin(score), npy.nanmax(score))
    plot_map_ax( axs[4],lon, lat, score,
        title='Combined Score',
        cmap='coolwarm',
        use_symlog=False
    )
    print('Min/max cluster map:', npy.nanmin(cluster_map), npy.nanmax(cluster_map))
    plot_map_ax( axs[5], lon,lat, cluster_map,
        title='Cluster Map',
        cmap='tab20',
        use_symlog=False
    )

    plt.tight_layout()

    plt.savefig('cluster_debug.png', dpi=300)
    plt.close(fig)