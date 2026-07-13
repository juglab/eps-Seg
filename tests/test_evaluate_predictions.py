import numpy as np
import tifffile as tiff

from eps_seg.evaluate_predictions import read_slice


def test_read_slice_returns_requested_tiff_page_without_base_stack(tmp_path):
    volume = np.arange(3 * 2 * 4, dtype=np.int8).reshape(3, 2, 4)
    path = tmp_path / "stack.tif"
    tiff.imwrite(path, volume, photometric="minisblack")

    image = read_slice(path, 1)

    np.testing.assert_array_equal(image, volume[1])
    assert image.base is None
