"""
Sub-pixel image registration by phase correlation with matrix-multiply DFT upsampling
(Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 33(2), 2008), plus synthetic wafer-pattern generators.

Dependencies: NumPy and SciPy only.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import scipy.ndimage as ndimage
from scipy.fft import fft2, ifft2, ifftshift


@dataclass
class RegistrationResult:
    shift_y: float   # shift (pixels) of `target` relative to `reference`, rows
    shift_x: float   # columns
    error: float     # 1 - normalised peak of the phase correlation (0 = perfect match)


# --------------------------------------------------------------------------- registration
def _upsampled_dft(data: np.ndarray, region: Tuple[int, int], upsample_factor: int,
                   offsets: np.ndarray) -> np.ndarray:
    """Upsampled inverse DFT of `data` on a small `region`, by two matrix products."""
    nor, noc = region
    nr, nc = data.shape
    col_kernel = np.exp(-2j * np.pi / (nc * upsample_factor)
                        * (ifftshift(np.arange(nc))[:, None] - np.floor(nc / 2))
                        * (np.arange(noc)[None, :] - offsets[1]))
    row_kernel = np.exp(-2j * np.pi / (nr * upsample_factor)
                        * (np.arange(nor)[:, None] - offsets[0])
                        * (ifftshift(np.arange(nr))[None, :] - np.floor(nr / 2)))
    return row_kernel @ data @ col_kernel


def register_subpixel_dft(reference: np.ndarray, target: np.ndarray,
                          upsample_factor: int = 100) -> RegistrationResult:
    """Estimate the (dy, dx) translation of `target` relative to `reference` to 1/upsample_factor px."""
    if reference.shape != target.shape:
        raise ValueError("reference and target must have the same shape")
    shape = np.array(reference.shape)
    reference = np.asarray(reference, dtype=np.float64)    # float64 keeps the peak search deterministic
    target = np.asarray(target, dtype=np.float64)
    product = fft2(target, workers=-1) * np.conj(fft2(reference, workers=-1))
    eps = np.finfo(np.float64).eps
    product /= np.maximum(np.abs(product), 100 * eps)            # phase correlation

    # 1. integer-pixel peak
    corr = ifft2(product, workers=-1)
    peak = np.array(np.unravel_index(np.argmax(np.abs(corr)), corr.shape), dtype=np.float64)
    peak[peak > shape // 2] -= shape[peak > shape // 2]
    if upsample_factor <= 1:
        return RegistrationResult(float(peak[0]), float(peak[1]), float(1 - np.abs(corr).max()))

    # 2. refine on a 1.5 x 1.5 px window around the peak at 1/upsample_factor px spacing
    shift = np.round(peak * upsample_factor) / upsample_factor
    region = int(np.ceil(upsample_factor * 1.5))
    centre = np.trunc(region / 2.0)
    fine = _upsampled_dft(product.conj(), (region, region), upsample_factor,
                          centre - shift * upsample_factor).conj()
    loc = np.array(np.unravel_index(np.argmax(np.abs(fine)), fine.shape), dtype=np.float64)
    shift += (loc - centre) / upsample_factor
    error = 1.0 - np.abs(fine).max() / product.size              # normalised peak height
    return RegistrationResult(float(shift[0]), float(shift[1]), float(error))


# --------------------------------------------------------------------------- shifting + noise
def apply_subpixel_shift_fourier(img: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Translate `img` by a fractional (dy, dx) with a Fourier phase ramp (circular, no interpolation)."""
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * shift_y + fx * shift_x))
    return np.real(ifft2(fft2(img, workers=-1) * ramp, workers=-1)).astype(np.float32)


def add_noise_snr(img: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio (dB, relative to mean signal power)."""
    noise_power = np.mean(img.astype(np.float64) ** 2) / 10 ** (snr_db / 10)
    return (img + rng.normal(0, np.sqrt(noise_power), img.shape)).astype(np.float32)


# --------------------------------------------------------------------------- synthetic wafers
def _psf(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma=sigma).astype(np.float32) if sigma > 0 else img


def generate_wafer_logic(shape=(512, 512), psf_sigma: float = 1.2) -> np.ndarray:
    """Logic layer: metal routing grid, dense contact vias, central crosshair fiducial."""
    h, w = shape
    img = np.zeros(shape, np.float32)
    img[::16, :] = 0.35
    img[:, ::16] = 0.35
    img[::8, ::8] = 0.75
    cy, cx = h // 2, w // 2
    img[cy - 35:cy + 35, cx - 3:cx + 3] = 1.0
    img[cy - 3:cy + 3, cx - 35:cx + 35] = 1.0
    return _psf(img, psf_sigma)


def generate_wafer_dram(shape=(512, 512), psf_sigma: float = 1.2) -> np.ndarray:
    """DRAM array: dense periodic capacitor matrix with a ring fiducial."""
    h, w = shape
    img = np.zeros(shape, np.float32)
    y, x = np.ogrid[:h, :w]
    img[(np.sin(2 * np.pi * y / 6) > 0.3) & (np.sin(2 * np.pi * x / 6) > 0.3)] = 0.70
    r = np.hypot(y - h // 2, x - w // 2)
    img[(r >= 60) & (r <= 68)] = 1.0
    return _psf(img, psf_sigma)


def generate_wafer_reticle(shape=(512, 512), psf_sigma: float = 1.2) -> np.ndarray:
    """Lithography overlay target: box-in-box with a central cross."""
    h, w = shape
    img = np.zeros(shape, np.float32)
    cy, cx = h // 2, w // 2
    img[cy - 50:cy + 50, cx - 50:cx - 42] = 1.0
    img[cy - 50:cy + 50, cx + 42:cx + 50] = 1.0
    img[cy - 50:cy - 42, cx - 50:cx + 50] = 1.0
    img[cy + 42:cy + 50, cx - 50:cx + 50] = 1.0
    img[cy - 20:cy + 20, cx - 2:cx + 2] = 0.90
    img[cy - 2:cy + 2, cx - 20:cx + 20] = 0.90
    return _psf(img, psf_sigma)


def generate_wafer_power_mems(shape=(512, 512), psf_sigma: float = 1.2) -> np.ndarray:
    """Power device / MEMS: wide interdigitated trench fingers (low spatial frequency)."""
    h, w = shape
    img = np.zeros(shape, np.float32)
    cy, cx = h // 2, w // 2
    for i in range(-120, 120, 24):
        img[cy + i:cy + i + 8, cx - 100:cx + 40] = 0.85
        img[cy + i + 12:cy + i + 20, cx - 40:cx + 100] = 0.85
    return _psf(img, psf_sigma)


WAFERS: Dict[str, Callable[..., np.ndarray]] = {
    "Logic": generate_wafer_logic,
    "DRAM array": generate_wafer_dram,
    "Overlay reticle": generate_wafer_reticle,
    "Power / MEMS": generate_wafer_power_mems,
}
