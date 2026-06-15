import numpy as np
import copy


class Shift:
    def __init__(self, f_shift, num_input):
        self.f_shift = f_shift
        self.num_input = num_input

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        # Shift events
        in_copy = copy.deepcopy(inp)
        for i in range(inp.shape[0]):
            in_copy[i,:,:] = 0.0
            s = np.random.randint(-self.f_shift, self.f_shift)
            if s < 0:
                in_copy[i,:,:s] = inp[i,:,-s:]
            elif s > 0:
                in_copy[i,:,s:] = inp[i,:,:-s]
        return in_copy

