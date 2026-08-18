import random

import numpy as np
import torch

from orbit.utils.seed import set_seed


def test_seed_determinism():
    # Run 1
    set_seed(1337, deterministic=True)
    py_rand_1 = [random.random() for _ in range(5)]
    np_rand_1 = np.random.rand(5).tolist()
    torch_rand_1 = torch.rand(5).tolist()

    # Run 2 with different seed
    set_seed(9999, deterministic=True)
    py_rand_diff = [random.random() for _ in range(5)]
    assert py_rand_1 != py_rand_diff

    # Run 3 with original seed
    set_seed(1337, deterministic=True)
    py_rand_2 = [random.random() for _ in range(5)]
    np_rand_2 = np.random.rand(5).tolist()
    torch_rand_2 = torch.rand(5).tolist()

    assert py_rand_1 == py_rand_2
    assert np_rand_1 == np_rand_2
    assert torch_rand_1 == torch_rand_2
