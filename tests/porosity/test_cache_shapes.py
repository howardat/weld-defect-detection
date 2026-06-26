import numpy as np

from experiments.build_cache import cache_to_matrices


def test_cache_to_matrices_shapes():
    cache = {
        "feature_names": ["mean", "std", "p10", "p50", "p90",
                          "contrast", "lap_var", "weld_area_frac",
                          "weld_aspect", "edge_density"],
        "param_order": ["block_size", "adapt_c", "open_ksize",
                        "erode_iters", "min_circularity", "darkness_thresh"],
        "items": [
            {"file_name": "a.jpg", "image_id": 0,
             "features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "params": {"block_size": 51, "adapt_c": 10, "open_ksize": 3,
                        "erode_iters": 5, "min_circularity": 0.3, "darkness_thresh": 20},
             "ceiling_f1": 0.8},
            {"file_name": "b.jpg", "image_id": 1,
             "features": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             "params": {"block_size": 31, "adapt_c": 20, "open_ksize": 5,
                        "erode_iters": 2, "min_circularity": 0.5, "darkness_thresh": 15},
             "ceiling_f1": 0.7},
        ],
    }
    X, Y = cache_to_matrices(cache)
    assert X.shape == (2, 10)
    assert Y.shape == (2, 6)
    assert Y[0, 0] == 51  # block_size of first item
