import numpy as np
from scipy.spatial.distance import pdist, squareform
import numba
from numba import cuda

pseudocount_weight_2p = 0.5
theta = 0.2
alphabet_order = "ACDEFGHIKLMNPQRSTVWY-"
q = len(alphabet_order)
_q = q - 1


class DCAScoreGen:
    def __init__(self, alignment_file):
        alignment, M, self.N = read_alignment(alignment_file)
        self.org_seq = alignment[0]
        Pi_true, Pij_true, M_eff = compute_statistics(alignment, M, self.N)
        pseudocount_weight_1p = 1 / M_eff
        Cij, Pi_small_reg = regularize_statistics(Pi_true, Pij_true, pseudocount_weight_1p, self.N)
        self.J, self.h = meanfield_inference(Cij, Pi_small_reg, self.N)
        self.reference_energy = compute_energy(self.J, self.h, self.org_seq, self.N)

    def score(self, sequence):
        seq_energy = compute_energy(self.J, self.h, sequence, self.N) - self.reference_energy
        return seq_energy

    def score_gpu(self, sequence_arr):
        v = cuda.to_device(np.zeros((len(sequence_arr), _q * self.N), dtype=np.uint16))
        energy_arr = cuda.to_device(np.full(len(sequence_arr), - self.reference_energy))
        J = cuda.to_device(self.J)
        h = cuda.to_device(self.h)
        compute_energy_gpu[256, 256](J, h, sequence_arr, v, self.N, _q, energy_arr)
        return energy_arr.copy_to_host()


# **********************************************************************************************************************

def read_alignment(fasta_file):
    print("Loading {}".format(fasta_file))
    with open(fasta_file) as ff:
        records = list(filter(None, ff.read().split(">")))
        sequences = [r.split("\n", 1)[1].replace("\n", "") for r in records]
    nr_of_seqs = len(sequences)
    mask = np.fromiter((alphabet_order.find(s) for s in sequences[0]), dtype=int) != -1
    nr_of_res = mask.sum()
    alignment = np.zeros((nr_of_seqs, nr_of_res), dtype=int)
    for idx, seq in enumerate(sequences):
        seq_as_nr = np.fromiter((alphabet_order.find(s) for i, s in enumerate(seq) if mask[i]), dtype=int,
                                count=nr_of_res)
        alignment[idx] = seq_as_nr
    print("Successfully loaded alignment with {} sequences, using {} residues.".format(nr_of_seqs, nr_of_res))
    return alignment, nr_of_seqs, nr_of_res


@numba.jit
def compute_statistics(alignment, M, N):
    W = 1. / (1 + squareform(pdist(alignment, metric="hamming") < theta).sum(axis=0))
    M_eff = W.sum()

    Pij_true = np.zeros((N * _q, N * _q))
    Pi_true = np.zeros(N * _q)

    for j in range(M):
        for i in range(N):
            a = alignment[j, i]
            if a != _q:
                Pi_true[a + i * _q] = Pi_true[a + i * _q] + W[j]
    Pi_true /= M_eff

    for l in range(M):
        for i in range(N - 1):
            a = alignment[l, i]
            if a != _q:
                for j in range(i + 1, N):
                    b = alignment[l, j]
                    if b != _q:
                        Pij_true[a + i * _q, b + j * _q] = Pij_true[a + i * _q, b + j * _q] + W[l]
                        Pij_true[b + j * _q, a + i * _q] = Pij_true[a + i * _q, b + j * _q]
    Pij_true /= M_eff
    for i in range(N):
        for a in range(_q):
            Pij_true[a + i * _q, a + i * _q] = Pi_true[a + i * _q]
    return Pi_true, Pij_true, M_eff


def regularize_statistics(Pi_true, Pij_true, pseudocount_weight_1p, N):
    Pi_small_reg = (1 - pseudocount_weight_1p) * Pi_true + pseudocount_weight_1p / q * np.ones(N * _q)
    pu = 1 / q / q * np.ones((N * _q, N * _q))
    for i in range(N):
        pu[mapkey(i, 0):mapkey(i, _q), mapkey(i, 0):mapkey(i, _q)] = 1 / q * np.identity(_q)
    pij = (1 - pseudocount_weight_2p) * Pij_true + pseudocount_weight_2p * pu
    pi = (1 - pseudocount_weight_2p) * Pi_true + pseudocount_weight_2p / q * np.ones(N * _q)
    Cij = pij - np.dot(pi, pi.transpose())
    return Cij, Pi_small_reg


def meanfield_inference(Cij, Pi_small_reg, N):
    Jgauss = -np.linalg.inv(Cij)
    J = np.copy(Jgauss)
    for i in range(N):
        J[mapkey(i, 0):mapkey(i, _q), mapkey(i, 0):mapkey(i, _q)] = np.zeros(_q)
    Pq_small_reg = np.zeros(mapkey(N - 1, _q))
    for i in range(N):
        Pq_small_reg[mapkey(i, 0):mapkey(i, _q)] = np.ones(_q) - sum(Pi_small_reg[mapkey(i, 0):mapkey(i, _q)])
    h = np.log(Pi_small_reg / Pq_small_reg.transpose()) - np.matmul(J, Pi_small_reg) \
        + 0.5 * np.diagonal(Jgauss) - np.matmul(Jgauss - J, Pi_small_reg)
    return J, h


def compute_energy(J, h, seq, N):
    v = np.zeros(N * _q)
    for i in range(N):
        if seq[i] < _q:
            v[mapkey(i, seq[i])] = 1
    energy = -0.5 * np.dot(np.dot(v.transpose(), J), v) - np.dot(v.transpose(), h)
    return energy


@cuda.jit
def compute_energy_gpu(J, h, seq_arr, v_global, n, _q, energy_arr):
    t = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if t < len(seq_arr):
        seq = seq_arr[t]
        v = v_global[t]
        pointer = 0
        for k in range(n):
            res = seq[k]
            if res < _q:
                v[pointer] = _q * k + res
                pointer += 1
        energy = 0
        for i in range(pointer):
            v_i = v[i]
            energy -= h[v_i]
            J_i = J[v_i]
            for k in range(pointer):
                v_k = v[k]
                energy -= .5 * J_i[v_k]
        energy_arr[t] += energy


def mapkey(i, alpha):
    return _q * i + alpha
