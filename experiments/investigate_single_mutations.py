import numpy as np
import os

from utils.mutation_generators import generate_single_mutations_generator
from utils.utilities import store_and_load
from utils.visualizations import visualize_single_mutations


@store_and_load
def investigate_single_mutations(seq, scoring_function):
    print("Calculating single mutation scores.")
    generate_single_mutations = generate_single_mutations_generator(seq)
    mutations = [mut for mut in generate_single_mutations()]
    sequences = []
    for pos, res in mutations:
        mutated_seq = seq.copy()
        mutated_seq[pos] = res
        sequences.append(mutated_seq)
    sequences = np.array(sequences)
    scores = scoring_function(sequences)
    m1_scores = {mutation: score for mutation, score in zip(mutations, scores)}
    return m1_scores


if __name__ == '__main__':
    from dca import DCAScoreGen
    import pickle

    tmp_path = "../tmp"
    tmp_dca_file = tmp_path + "/dca.pickle"
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    if os.path.isfile(tmp_dca_file):
        print("Loading DCA model from file")
        with open(tmp_dca_file, "rb") as f:
            dca = pickle.load(f)
    else:
        print("Calculating DCA model")
        dca = DCAScoreGen("../sample_data/9C542562-5C10-11E7-AF2C-EEEADBC3747A.3.afa")
        with open(tmp_dca_file, "wb") as f:
            pickle.dump(dca, f)

    m1_scores = investigate_single_mutations(dca.org_seq, dca.score_gpu)
    visualize_single_mutations(m1_scores, "../plots/single_mutations.png")
