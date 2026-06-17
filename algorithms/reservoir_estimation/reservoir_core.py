from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Lightweight, independent Nurek A-H-V curve for a basic delivery tool.
# Columns are: water_level_m, area_km2, volume_mcm.
# Source basis: median grouped legacy AW3D30/Sentinel-2 prototype points from
# E:\Rce\data\external\legacy_aw3d30_volume_points.csv. This file intentionally
# embeds the curve so the tool does not depend on the paper workflow.
DEFAULT_CURVE = np.array(
    [
        (868.0, 50.81860, 1514.674784),
        (869.0, 53.64320, 1781.539013),
        (870.0, 53.15735, 1733.599270),
        (871.0, 53.67795, 1786.218810),
        (872.0, 53.54780, 1772.050947),
        (874.0, 54.78975, 1898.451753),
        (875.0, 56.72260, 2106.297412),
        (876.0, 56.72730, 2107.022159),
        (877.0, 56.63660, 2096.731538),
        (878.0, 56.82280, 2117.479412),
        (879.0, 59.16410, 2390.155152),
        (880.0, 56.70030, 2103.814171),
        (881.0, 59.79550, 2467.498010),
        (882.0, 61.49865, 2684.405086),
        (883.0, 61.92345, 2740.454989),
        (884.0, 62.12310, 2767.024368),
        (885.0, 62.55620, 2825.285784),
        (886.0, 62.78265, 2856.198302),
        (887.0, 62.90990, 2873.480739),
        (888.0, 63.78780, 2995.464503),
        (889.0, 63.06050, 2894.166655),
        (890.0, 63.75750, 2991.197881),
        (891.0, 64.52590, 3101.033453),
        (892.0, 65.75930, 3281.881158),
        (893.0, 65.27935, 3211.379070),
        (894.0, 65.39000, 3226.898595),
        (895.0, 65.41600, 3230.749309),
        (896.0, 65.77990, 3284.966406),
        (896.5, 66.05180, 3325.870060),
        (897.0, 66.35250, 3371.500107),
        (898.0, 67.57100, 3560.748288),
        (900.0, 68.15835, 3655.159579),
        (901.0, 68.58910, 3724.059564),
        (902.0, 69.08410, 3805.271236),
        (904.0, 69.51540, 3876.987991),
        (905.0, 69.55550, 3883.700550),
        (906.0, 70.09500, 3974.774380),
        (907.0, 70.35780, 4019.648330),
        (908.0, 71.13725, 4154.735479),
        (909.0, 70.92260, 4117.231159),
        (910.0, 71.85290, 4281.384621),
        (911.0, 72.02845, 4312.902931),
        (912.0, 72.33605, 4368.455977),
        (913.0, 72.17880, 4339.905553),
        (914.0, 72.87140, 4466.041704),
        (915.0, 74.81620, 4833.238699),
        (916.0, 75.39485, 4947.595457),
        (917.0, 75.95940, 5058.198665),
        (918.0, 75.81730, 5029.864072),
        (919.0, 76.59890, 5187.031732),
        (920.0, 76.98915, 5266.726220),
    ],
    dtype=float,
)


@dataclass
class EstimateResult:
    date: str
    input_type: str
    water_level_m: float | None
    area_km2: float | None
    estimated_volume_mcm: float
    estimated_volume_km3: float
    model_area_km2: float | None
    model_level_m: float | None
    total_capacity_percent: float
    active_storage_percent: float
    method: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            f"Date: {self.date or 'not provided'}",
            f"Input type: {self.input_type}",
            f"Estimated volume: {self.estimated_volume_mcm:.1f} MCM ({self.estimated_volume_km3:.3f} km3)",
            f"Total capacity reference: {self.total_capacity_percent:.1f}% of 10.5 km3",
            f"Active storage reference: {self.active_storage_percent:.1f}% of 4.2 km3",
            f"Method: {self.method}",
        ]
        if self.water_level_m is not None:
            lines.append(f"Input/estimated water level: {self.water_level_m:.2f} m")
        if self.area_km2 is not None:
            lines.append(f"Input/estimated water area: {self.area_km2:.2f} km2")
        if self.model_area_km2 is not None:
            lines.append(f"Curve area at level: {self.model_area_km2:.2f} km2")
        if self.model_level_m is not None:
            lines.append(f"Curve level from area: {self.model_level_m:.2f} m")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines)


@dataclass
class ImageAreaResult:
    image_path: str
    area_km2: float
    water_pixels: int
    total_pixels: int
    pixel_area_m2: float
    threshold: float
    water_mode: str
    method: str
    warnings: list[str]


class NurekReservoirEstimator:
    def __init__(
        self,
        curve: np.ndarray | None = None,
        total_capacity_km3: float = 10.5,
        active_storage_km3: float = 4.2,
    ) -> None:
        self.total_capacity_km3 = float(total_capacity_km3)
        self.active_storage_km3 = float(active_storage_km3)
        self.curve = self._prepare_curve(DEFAULT_CURVE if curve is None else curve)

    @staticmethod
    def _prepare_curve(curve: np.ndarray) -> np.ndarray:
        arr = np.asarray(curve, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Curve must have columns: water_level_m, area_km2, volume_mcm")
        arr = arr[np.argsort(arr[:, 0])].copy()
        # Enforce monotonicity for simple interpolation. The source curve is a
        # prototype and contains small same-level/DEM-noise reversals.
        arr[:, 1] = np.maximum.accumulate(arr[:, 1])
        arr[:, 2] = np.maximum.accumulate(arr[:, 2])
        return arr

    @property
    def level_range(self) -> tuple[float, float]:
        return float(self.curve[0, 0]), float(self.curve[-1, 0])

    @property
    def area_range(self) -> tuple[float, float]:
        return float(self.curve[0, 1]), float(self.curve[-1, 1])

    def estimate_manual(
        self,
        date: str = "",
        water_level_m: float | None = None,
        area_km2: float | None = None,
    ) -> EstimateResult:
        if water_level_m is None and area_km2 is None:
            raise ValueError("Provide at least water_level_m or area_km2.")
        if water_level_m is not None:
            result = self.estimate_from_level(date=date, water_level_m=water_level_m)
            if area_km2 is not None:
                result.area_km2 = float(area_km2)
                if result.model_area_km2 is not None:
                    delta = abs(float(area_km2) - result.model_area_km2)
                    if delta > 5.0:
                        result.warnings.append(
                            f"Manual area differs from curve area by {delta:.2f} km2; level-based volume is used."
                        )
            return result
        return self.estimate_from_area(date=date, area_km2=float(area_km2))

    def estimate_from_level(self, date: str, water_level_m: float) -> EstimateResult:
        warnings = self._range_warning(float(water_level_m), self.level_range, "water level", "m")
        volume = self._interp_or_extrapolate(float(water_level_m), self.curve[:, 0], self.curve[:, 2])
        area = self._interp_or_extrapolate(float(water_level_m), self.curve[:, 0], self.curve[:, 1])
        return self._make_result(
            date=date,
            input_type="manual_water_level",
            water_level_m=float(water_level_m),
            area_km2=None,
            volume_mcm=volume,
            model_area_km2=area,
            model_level_m=None,
            method="Level-to-volume linear interpolation on embedded Nurek prototype curve.",
            warnings=warnings,
        )

    def estimate_from_area(self, date: str, area_km2: float) -> EstimateResult:
        warnings = self._range_warning(float(area_km2), self.area_range, "water area", "km2")
        level = self._interp_or_extrapolate(float(area_km2), self.curve[:, 1], self.curve[:, 0])
        volume = self._interp_or_extrapolate(float(area_km2), self.curve[:, 1], self.curve[:, 2])
        return self._make_result(
            date=date,
            input_type="manual_area_or_image_area",
            water_level_m=level,
            area_km2=float(area_km2),
            volume_mcm=volume,
            model_area_km2=None,
            model_level_m=level,
            method="Area-to-level and area-to-volume linear interpolation on embedded Nurek prototype curve.",
            warnings=warnings,
        )

    def estimate_from_image(
        self,
        image_path: str | Path,
        date: str = "",
        pixel_size_m: float | None = None,
        threshold: float | None = None,
        water_mode: str = "dark",
    ) -> tuple[ImageAreaResult, EstimateResult]:
        area_result = estimate_water_area_from_image(
            image_path=image_path,
            pixel_size_m=pixel_size_m,
            threshold=threshold,
            water_mode=water_mode,
        )
        estimate = self.estimate_from_area(date=date, area_km2=area_result.area_km2)
        estimate.input_type = "uploaded_remote_sensing_image"
        estimate.method = (
            "Image threshold water-area extraction, then area-to-volume interpolation. "
            "Use only as a basic visualization estimate."
        )
        estimate.warnings.extend(area_result.warnings)
        return area_result, estimate

    def _make_result(
        self,
        date: str,
        input_type: str,
        water_level_m: float | None,
        area_km2: float | None,
        volume_mcm: float,
        model_area_km2: float | None,
        model_level_m: float | None,
        method: str,
        warnings: list[str],
    ) -> EstimateResult:
        volume_km3 = volume_mcm / 1000.0
        return EstimateResult(
            date=date,
            input_type=input_type,
            water_level_m=water_level_m,
            area_km2=area_km2,
            estimated_volume_mcm=volume_mcm,
            estimated_volume_km3=volume_km3,
            model_area_km2=model_area_km2,
            model_level_m=model_level_m,
            total_capacity_percent=100.0 * volume_km3 / self.total_capacity_km3,
            active_storage_percent=100.0 * volume_km3 / self.active_storage_km3,
            method=method,
            warnings=warnings,
        )

    @staticmethod
    def _range_warning(value: float, valid_range: tuple[float, float], label: str, unit: str) -> list[str]:
        lo, hi = valid_range
        if value < lo or value > hi:
            return [
                f"Input {label} {value:.2f} {unit} is outside calibration range "
                f"{lo:.2f}-{hi:.2f} {unit}; result is extrapolated."
            ]
        return []

    @staticmethod
    def _interp_or_extrapolate(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        if x < xp[0]:
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            return float(fp[0] + slope * (x - xp[0]))
        if x > xp[-1]:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return float(fp[-1] + slope * (x - xp[-1]))
        return float(np.interp(x, xp, fp))


def estimate_water_area_from_image(
    image_path: str | Path,
    pixel_size_m: float | None = None,
    threshold: float | None = None,
    water_mode: str = "dark",
) -> ImageAreaResult:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    gray, pixel_area_m2, read_warnings = _read_gray_image(path, pixel_size_m=pixel_size_m)
    gray = np.asarray(gray, dtype=float)
    valid = np.isfinite(gray)
    if not valid.any():
        raise ValueError("Image has no valid pixels.")
    values = gray[valid]
    if values.max() > 1.5:
        gray = gray / 255.0
        values = gray[valid]

    used_threshold = otsu_threshold(values) if threshold is None else float(threshold)
    mode = water_mode.lower().strip()
    if mode not in {"dark", "bright"}:
        raise ValueError("water_mode must be 'dark' or 'bright'.")
    water = (gray <= used_threshold) if mode == "dark" else (gray >= used_threshold)
    water &= valid

    water_pixels = int(water.sum())
    area_km2 = water_pixels * pixel_area_m2 / 1e6
    warnings = list(read_warnings)
    if water_pixels == 0:
        warnings.append("No water pixels were detected. Adjust threshold or water mode.")
    if water_pixels == int(valid.sum()):
        warnings.append("All valid pixels were classified as water. Adjust threshold or crop the image.")
    return ImageAreaResult(
        image_path=str(path),
        area_km2=float(area_km2),
        water_pixels=water_pixels,
        total_pixels=int(valid.sum()),
        pixel_area_m2=float(pixel_area_m2),
        threshold=float(used_threshold),
        water_mode=mode,
        method="grayscale_threshold_otsu" if threshold is None else "grayscale_threshold_manual",
        warnings=warnings,
    )


def _read_gray_image(path: Path, pixel_size_m: float | None) -> tuple[np.ndarray, float, list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                arr = src.read()
                image = _bands_to_gray(arr)
                pixel_area = abs(src.transform.a * src.transform.e)
                if not np.isfinite(pixel_area) or pixel_area <= 0:
                    pixel_area = _pixel_area_from_user(pixel_size_m)
                    warnings.append("GeoTIFF transform is unavailable; using manual pixel size.")
                return image, float(pixel_area), warnings
        except Exception as exc:
            warnings.append(f"GeoTIFF read fallback used: {exc}")

    try:
        import matplotlib.image as mpimg

        image = mpimg.imread(path)
    except Exception as exc:
        raise ValueError(f"Cannot read image: {path}") from exc
    pixel_area = _pixel_area_from_user(pixel_size_m)
    warnings.append("Non-GeoTIFF image: area uses the manual pixel size.")
    return _bands_to_gray(image), pixel_area, warnings


def _pixel_area_from_user(pixel_size_m: float | None) -> float:
    if pixel_size_m is None or float(pixel_size_m) <= 0:
        raise ValueError("A positive pixel_size_m is required for non-georeferenced images.")
    return float(pixel_size_m) ** 2


def _bands_to_gray(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 2:
        return data.astype(float)
    if data.ndim == 3 and data.shape[0] in {1, 2, 3, 4}:
        data = np.moveaxis(data, 0, -1)
    if data.ndim == 3:
        channels = data[..., :3].astype(float)
        if channels.shape[-1] == 1:
            return channels[..., 0]
        return 0.299 * channels[..., 0] + 0.587 * channels[..., 1] + 0.114 * channels[..., 2]
    raise ValueError(f"Unsupported image dimensions: {data.shape}")


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No valid values for thresholding.")
    hist, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    sum_total = np.dot(hist, centers)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~np.isfinite(between)] = -1
    return float(centers[int(np.argmax(between))])


def save_curve_plot(result: EstimateResult, output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = NurekReservoirEstimator()
    curve = estimator.curve
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    ax.plot(curve[:, 0], curve[:, 2], color="#1f6f8b", linewidth=2.0, label="Embedded Nurek curve")
    if result.water_level_m is not None:
        ax.scatter(
            [result.water_level_m],
            [result.estimated_volume_mcm],
            color="#d1495b",
            s=55,
            zorder=5,
            label="Current estimate",
        )
    ax.set_xlabel("Water level (m)")
    ax.set_ylabel("Estimated volume (MCM)")
    ax.set_title("Nurek Reservoir Basic Storage Estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    estimator = NurekReservoirEstimator()
    demo = estimator.estimate_manual(date="2026-06-17", water_level_m=900.0)
    print(demo.summary_text())
