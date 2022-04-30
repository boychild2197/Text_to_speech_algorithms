from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra


def to_node_index(i: int, j: int, cols: int) -> int:
    return cols * i + j


def from_node_index(node_index: int, cols: int) -> Tuple[int, int]:
    return node_index // cols, node_index % cols


def to_adj_matrix(mat: np.array) -> csr_matrix:
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)

    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()


def extract_durations_with_dijkstra(seq: np.array,
                                    att: np.array,
                                    mel_len: int) -> np.array:
    """
    Extracts durations from the attention matrix by finding the shortest monotonic path from
    top left to bottom right.
    """

    path_probs = 1.-att[:mel_len, :]
    adj_matrix = to_adj_matrix(path_probs)
    dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True,
                                         indices=0, return_predecessors=True)
    path = []
    pr_index = predecessors[-1]
    while pr_index != 0:
        path.append(pr_index)
        pr_index = predecessors[pr_index]
    path.reverse()

    # append first and last node
    path = [0] + path + [dist_matrix.size-1]
    cols = path_probs.shape[1]
    mel_text = {}
    durations = np.zeros(seq.shape[0], dtype=np.int32)

    # collect indices (mel, text) along the path
    for node_index in path:
        i, j = from_node_index(node_index, cols)
        mel_text[i] = j

    for j in mel_text.values():
        durations[j] += 1

    return durations


def extract_durations_per_count(seq: np.array,
                                att: np.array,
                                mel_len: int) -> np.array:
    argmax = np.argmax(att[:, :], axis=1)
    durations = np.zeros(seq.shape[0], dtype=np.int32)
    # fix random jumps in attention
    for j in range(1, argmax.shape[0]):
        if abs(argmax[j] - argmax[j - 1]) > 10:
            argmax[j] = argmax[j - 1]
    count = np.bincount(argmax[:mel_len])
    durations[:len(count)] = count[:len(count)]
    return durations
