# distutils: language = c++
import numpy as np
cimport numpy as np


def mvp(mat, vec, wires, num_wires):
    r"""Matrix-vector product to be wrapped in LightningQubit method.

    Args:
        mat (array): matrix to multiply
        vec (array): state vector to multiply
        wires (Sequence[int]): target subsystems
        num_wires: total number of wires in circuit

    Returns:
        array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
    """
    # TODO: use multi-index vectors/matrices to represent states/gates internally
    # mat = np.reshape(mat, [2] * len(wires) * 2)
    vec = np.reshape(vec, [2] * num_wires)
    axes = (np.arange(len(wires), 2 * len(wires)), wires)
    tdot = np.tensordot(mat, vec, axes=axes)

    # tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [idx for idx in range(num_wires) if idx not in wires]
    perm = wires + unused_idxs
    inv_perm = np.argsort(perm)  # argsort gives inverse permutation
    state_multi_index = np.transpose(tdot, inv_perm)
    return np.reshape(state_multi_index, 2 ** num_wires)
