import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
import contextily as ctx
import pyproj

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Needed for generate_output() calls
from functions.model_object import generate_output

# -------------------------
# Reverse geocoding helpers
# -------------------------
_geolocator = Nominatim(user_agent="shoreline-model")
_reverse = RateLimiter(_geolocator.reverse, min_delay_seconds=1.0)

def best_place_name(lat, lon):
    """
    Try to extract a beach or locality style name from lat/lon.
    Falls back to a simple display_name head if specific keys not found.
    No caching by request.
    """
    try:
        loc = _reverse((lat, lon), language="en", addressdetails=True, zoom=15)
        if not loc or "address" not in getattr(loc, "raw", {}):
            return None
        addr = loc.raw["address"]

        # Prefer coastal features first
        for key in ["beach", "bay", "harbour", "island", "coast"]:
            if key in addr:
                return addr[key]

        # Then settlement hierarchy
        for key in ["suburb", "locality", "neighbourhood", "town", "city"]:
            if key in addr:
                return addr[key]

        # Last resort: take the first token of display_name
        return str(loc).split(",")[0]
    except Exception:
        return None


def compare_outputs(transect, model, data, settings):
    t1 = transect
    unseen_target = [t1]

    # ---------- model outputs ----------
    static_ablation_output, _ = generate_output(unseen_target, model, data, settings, ablation_type='static')
    signal_ablation_output, _ = generate_output(unseen_target, model, data, settings, ablation_type='signal')
    partial_signal_ablation_output, _ = generate_output(unseen_target, model, data, settings, ablation_type='signal_partial')
    dynamic_ablation_output, _ = generate_output(unseen_target, model, data, settings, ablation_type='dynamic')
    gxt_output, gxt_scores = generate_output(unseen_target, model, data, settings)

    # ---------- colors ----------
    best_col   = '#2e2e2e'
    signal_col = '#0c64ea'   # blue
    partial_signal_col = "#961ed6"
    static_col = '#CC0066'
    dynamic_col = '#0d8c62' 

    # ---------- figure + layout ----------
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 2.0])

    # ---------- projections ----------
    fwd_proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    inv_proj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    # ---------- load transect and reproject once ----------
    CoastSat_transects = gpd.read_file('data/geojson/CoastSat_transect_layer_NSW.geojson').set_index('TransectId')
    transects_subset = CoastSat_transects.loc[[t1]].to_crs(epsg=3857)

    # ---------- map axis with meter-based bbox ----------
    ax_map = fig.add_subplot(gs[:, 0])

    # centroid in Web Mercator
    centroid = transects_subset.geometry.iloc[0].centroid
    cx, cy = centroid.x, centroid.y

    # bbox in meters
    dx, dy = 1500, 3000  # ~1.5 km x 3 km window
    x_min, x_max = cx - dx, cx + dx
    y_min, y_max = cy - dy, cy + dy
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)

    # plot transect
    transects_subset.plot(ax=ax_map, color='w', linewidth=3, zorder=10)

    # reverse geocode for name
    lon, lat = inv_proj(cx, cy)
    place_name = best_place_name(lat, lon) or f"Transect {t1}"

    # label near the transect
    ax_map.text(
        cx + 650, cy - 350, f"{t1}",
        fontsize=11, color='white', fontweight='bold',
        ha='center', va='center', zorder=11, bbox=dict(facecolor='black', alpha=0, lw=0)
    )

    # basemap
    ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, attribution=False)

    # format ticks as lon/lat using midpoints
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    def format_lon(x, _):
        lon_val, _ = inv_proj(x, y_mid)
        return f"{lon_val:.3f}"

    def format_lat(y, _):
        _, lat_val = inv_proj(x_mid, y)
        return f"{lat_val:.3f}"

    ax_map.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax_map.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    ax_map.set_xlabel("Longitude", fontsize=12)
    ax_map.set_ylabel("Latitude", fontsize=12)
    ax_map.grid(False)

    # ---------- time series axes ----------
    ax_ts1 = fig.add_subplot(gs[0, 1])
    ax_ts2 = fig.add_subplot(gs[1, 1])
    ax_ts3 = fig.add_subplot(gs[2, 1])

    for ax in [ax_ts1, ax_ts2, ax_ts3]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Shoreline (m)", fontsize=11)

    # ground truth once
    ground_truth = data.df.loc[data.holdout.index[-settings['output_length']:]].copy()
    gt_series = ground_truth[t1].dropna()

    # common scatter + faint line
    for ax in [ax_ts1, ax_ts2, ax_ts3]:
        ax.scatter(gt_series.index, gt_series, color='k', facecolor='w', alpha=1, s=10, zorder=1, marker='s')
        ax.plot(gt_series, color='grey', alpha=0.5, zorder=0, lw=1)

    # TS1: best model + static ablation
    ax_ts1.plot(ground_truth.index, gxt_output[t1][:, 1], color=best_col, lw=1.5, label='Model Output')
    ax_ts1.fill_between(ground_truth.index, gxt_output[t1][:, 0], gxt_output[t1][:, 2], lw=0, color=best_col, alpha=0.1)
    ax_ts1.plot(ground_truth.index, static_ablation_output[t1][:, 1], color=static_col, lw=1.5, label='Static Ablation')
    ax_ts1.set_title(f"{place_name} ({t1})")

    # TS2: best model + signal ablations
    ax_ts2.plot(ground_truth.index, gxt_output[t1][:, 1], color=best_col, lw=1.5)
    ax_ts2.fill_between(ground_truth.index, gxt_output[t1][:, 0], gxt_output[t1][:, 2], lw=0, color=best_col, alpha=0.1)
    ax_ts2.plot(ground_truth.index, signal_ablation_output[t1][:, 1], color=signal_col, lw=1.5, label='$y_{prev}$ Ablation')
    ax_ts2.plot(ground_truth.index, partial_signal_ablation_output[t1][:, 1], color=partial_signal_col, lw=1.5, label='$y_{prev}$ Ablation (Partial)')

    # TS3: best model + dynamic ablation
    ax_ts3.plot(ground_truth.index, gxt_output[t1][:, 1], color=best_col, lw=1.5)
    ax_ts3.fill_between(ground_truth.index, gxt_output[t1][:, 0], gxt_output[t1][:, 2], lw=0, color=best_col, alpha=0.1)
    ax_ts3.plot(ground_truth.index, dynamic_ablation_output[t1][:, 1], color=dynamic_col, lw=1.5, label='Dynamic Ablation')

    # styling
    for ax in [ax_map, ax_ts1, ax_ts2, ax_ts3]:
        ax.set_facecolor('#fafafa')
        ax.grid(linestyle='--', linewidth=.2, axis='both', zorder=0)
        ax.tick_params(axis='both', which='major', color='gray', direction='out',
                       top=True, bottom=True, left=True, right=True, labelsize=10)
        ax.tick_params(axis='both', which='minor', color='gray', direction='out',
                       top=True, bottom=True, left=True, right=True)

    # shared x-lims
    for ax in [ax_ts1, ax_ts2, ax_ts3]:
        ax.set_xlim(ground_truth.index[0], ground_truth.index[-1])

    # legend from all three
    handles1, labels1 = ax_ts1.get_legend_handles_labels()
    handles2, labels2 = ax_ts2.get_legend_handles_labels()
    handles3, labels3 = ax_ts3.get_legend_handles_labels()
    handles = handles1 + handles2 + handles3
    labels = labels1 + labels2 + labels3

    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(unique.values(), unique.keys(),
               loc='lower center', bbox_to_anchor=(0.7, -0.03),
               ncol=3, fontsize=12, frameon=False)

    ax_map.grid(False)

    # leave room for legend at bottom
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

#####################
#####################
