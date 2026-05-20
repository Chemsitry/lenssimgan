"""Load real/target images and simulated lens images for SimGAN training.

Two target-image sources are supported:
  - 'jwst': Real GOODS-S F115W cutouts from the JADES catalog. Can include
    blank/noisy frames that confuse the discriminator.
  - 'vela': VELA cosmological simulation images (JWST NIRCam F115W mock
    observations). Cleaner and more consistent than raw JWST cutouts.

Simulated images come from the existing lenstronomy-based simulator output
in /global/cfs/projectdirs/deepsrch.

All images are normalised per-image to [0, 1] (min-max).
"""

import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

JWST_IMAGE_PATH = "/global/cfs/projectdirs/deepsrch/jwst_sims/data/JWST/goods_s_F115W_2018_08_29.fits"
JADES_CATALOG_PATH = "/global/cfs/projectdirs/deepsrch/jwst_sims/data/JWST/JADES_SF_mock_r1_v1.1.fits"
SIM_DIR = "/global/cfs/projectdirs/deepsrch/jwst_sims/sims.15-.5"

VELA_DIR = "/global/cfs/projectdirs/deepsrch/vela"
VELA_MIN_MEAN = 1e-5  # skip images whose mean pixel value is below this (blank frames)

CUTOUT_SIZE = 125  # pixels — matches the simulator output


def _normalise_per_image(stack: np.ndarray) -> np.ndarray:
    """Min-max normalise each image independently to [0, 1]."""
    out = np.empty_like(stack, dtype=np.float32)
    for i in range(len(stack)):
        img = stack[i].astype(np.float32)
        lo, hi = np.nanmin(img), np.nanmax(img)
        if hi > lo and np.isfinite(hi) and np.isfinite(lo):
            out[i] = (img - lo) / (hi - lo)
        else:
            out[i] = 0.0
    return out


def load_real_jwst_cutouts(max_cutouts: int = 5000, stride: int = 1) -> np.ndarray:
    """Cut 125x125 postage stamps from the GOODS-S F115W mosaic.

    `stride` lets you sub-sample the catalog to spread cutouts across
    the field instead of taking the first N consecutive entries.
    Returns float32 array of shape (N, 125, 125), values in [0, 1].
    """
    with fits.open(JWST_IMAGE_PATH) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        image_data = hdu.data.astype(np.float32)

    with fits.open(JADES_CATALOG_PATH) as cat_hdul:
        cat = cat_hdul[1].data
        ras = np.asarray(cat["RA"])
        decs = np.asarray(cat["DEC"])

    cutouts = []
    for ra, dec in zip(ras[::stride], decs[::stride]):
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        try:
            cut = Cutout2D(image_data, coord, CUTOUT_SIZE, wcs=wcs, mode="strict")
        except Exception:
            continue
        if cut.data.shape == (CUTOUT_SIZE, CUTOUT_SIZE) and np.isfinite(cut.data).all():
            cutouts.append(cut.data)
        if len(cutouts) >= max_cutouts:
            break

    if not cutouts:
        raise RuntimeError(
            "No valid JWST cutouts produced. Check that the catalog "
            "RA/DEC actually overlap the GOODS-S image footprint."
        )

    arr = np.stack(cutouts, axis=0)
    return _normalise_per_image(arr)


def load_simulated_images(lens_only: bool = True, max_images: int | None = None) -> np.ndarray:
    """Load the simulator output. Returns float32, shape (N, 125, 125), in [0, 1]."""
    images = np.load(f"{SIM_DIR}/images.npy", mmap_mode="r")
    labels = np.load(f"{SIM_DIR}/lensed.npy")
    if lens_only:
        keep = labels == 1
        images = np.array(images[keep])
    else:
        images = np.array(images)
    if max_images is not None:
        images = images[:max_images]
    return _normalise_per_image(images)


def load_vela_images(
    max_images: int | None = None,
    filter_band: str = "f115w",
    min_mean: float = VELA_MIN_MEAN,
) -> np.ndarray:
    """Load VELA cosmological simulation images as the target distribution.

    VELA images are mock JWST NIRCam observations of simulated galaxies.
    Each FITS file is 800x800 pixels; we take the central 125x125 crop
    because VELA galaxies are centred in their frame.

    Blank frames (mean pixel value below min_mean) are skipped — they
    contain no useful galaxy signal and would teach the discriminator
    to treat blank images as "real".

    Returns float32 array of shape (N, 125, 125), values in [0, 1].
    """
    pattern = f"{VELA_DIR}/vela*/cam*/jwst/nircam/{filter_band}/*.fits"
    all_paths = sorted(glob.glob(pattern))
    if not all_paths:
        raise RuntimeError(
            f"No VELA FITS files found matching {pattern}. "
            "Check that the VELA directory is accessible."
        )

    rng = np.random.default_rng(seed=0)
    rng.shuffle(all_paths)

    half = CUTOUT_SIZE // 2
    centre = 400  # centre of the 800x800 VELA frame
    crop_slice = slice(centre - half, centre - half + CUTOUT_SIZE)

    cutouts = []
    for path in all_paths:
        try:
            with fits.open(path, memmap=True) as hdul:
                data = hdul["IMAGE_PRISTINE"].data.astype(np.float32)
        except Exception:
            continue
        crop = data[crop_slice, crop_slice]
        if crop.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
            continue
        if not np.isfinite(crop).all():
            continue
        if crop.mean() < min_mean:
            continue
        cutouts.append(crop)
        if max_images is not None and len(cutouts) >= max_images:
            break

    if not cutouts:
        raise RuntimeError(
            "No usable VELA images found (all were blank or had bad pixels). "
            f"Try lowering min_mean (currently {min_mean})."
        )

    arr = np.stack(cutouts, axis=0)
    return _normalise_per_image(arr)


if __name__ == "__main__":
    # Smoke test: load a small batch of each and report shapes/stats.
    print("Loading 32 real JWST cutouts...")
    real = load_real_jwst_cutouts(max_cutouts=32, stride=500)
    print(f"  shape={real.shape}, min={real.min():.3f}, max={real.max():.3f}, mean={real.mean():.3f}")

    print("Loading 32 simulated images...")
    sim = load_simulated_images(max_images=32)
    print(f"  shape={sim.shape}, min={sim.min():.3f}, max={sim.max():.3f}, mean={sim.mean():.3f}")

    print("Loading 32 VELA images...")
    vela = load_vela_images(max_images=32)
    print(f"  shape={vela.shape}, min={vela.min():.3f}, max={vela.max():.3f}, mean={vela.mean():.3f}")
