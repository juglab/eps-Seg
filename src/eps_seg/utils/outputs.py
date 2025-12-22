from pathlib import Path
import zarr
import tifffile


def zarr_to_tiff(
    zarr_path: str | Path,
    *,
    compression: str | None = "zstd",
    tile: tuple[int, int, int] | None = None,
):
    """
    Convert a plain Zarr array to a BigTIFF file.

    Parameters
    ----------
    zarr_path : str or Path
        Path to a .zarr directory containing a single Zarr array.
    compression : str or None
        TIFF compression (e.g. "zstd", "lz4", None).
    tile : tuple or None
        TIFF tile shape as (Z, Y, X). If None, a reasonable default is chosen.
    """

    zarr_path = Path(zarr_path)
    if zarr_path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got: {zarr_path}")

    tif_path = zarr_path.with_suffix(".tif")

    z = zarr.open(zarr_path, mode="r")

    if z.ndim != 3:
        raise ValueError(
            f"Expected a 3D Zarr array (Z,Y,X), got shape {z.shape}"
        )

    print(f"Converting {zarr_path} → {tif_path}")
    print(f"Shape: {z.shape}, dtype: {z.dtype}, tile: {tile}")

    with tifffile.TiffWriter(tif_path, bigtiff=True) as tif:
        tif.write(
            z,
            shape=z.shape,
            dtype=z.dtype,
        )

    print(f"Saved TIFF: {tif_path}")
