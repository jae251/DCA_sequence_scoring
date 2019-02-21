import numpy as np
import os

from utils.mutation_generators import BiasedRandomMutationsGenerator
from utils.utilities import store_and_load
from investigate_single_mutations import investigate_single_mutations
from utils.visualizations import visualize_score_by_mutation_nr


@store_and_load
def investigate_all_combinations_of_beneficial_single_mutations(seq, scoring_function, single_mutations_scores,
                                                                mutation_pool_cutoff=None,
                                                                max_nr_of_mutations_per_seq=None,
                                                                max_nr_of_seq=None):
    print("Calculating biased combination trial")
    generate_mutations = BiasedRandomMutationsGenerator(seq, single_mutations_scores, mutation_pool_cutoff,
                                                        max_nr_of_mutations_per_seq, max_nr_of_seq)
    results = []
    sequences = []
    nr_of_mutations = []
    for i, mutated_seq in enumerate(generate_mutations):
        nr_of_mutations.append(np.sum(seq != mutated_seq))
        sequences.append(mutated_seq)
        if i % 65535 == 0 and i != 0:  # using 256 grids with 256 threads on GPU
            sequences = np.array(sequences)
            scores = scoring_function(sequences)
            results = np.append(results, scores)
            sequences = []
    sequences = np.array(sequences)
    scores = scoring_function(sequences)
    results = np.append(results, scores)
    nr_of_mutations = np.array(nr_of_mutations)
    return results, nr_of_mutations


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
    results, nr_of_mutations = investigate_all_combinations_of_beneficial_single_mutations(dca.org_seq, dca.score_gpu, m1_scores)

    # appending single mutation results for visu
    m1_scores = np.array(list(m1_scores.values()))
    m1_scores = m1_scores[m1_scores <= 0]
    results = np.append(results, m1_scores)
    nr_of_mutations = np.append(nr_of_mutations, np.ones(len(m1_scores)))

    visualize_score_by_mutation_nr(results, nr_of_mutations, "../plots/best_mutation_combinations.png")
