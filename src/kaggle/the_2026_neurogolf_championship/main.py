import numpy as np

def fits_rotation90(examples):
    for ex in examples:
        if not np.array_equal(
            np.rot90(ex["input"]),
            ex["output"]
        ):
            return False
    return True