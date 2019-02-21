import numpy as np
import os

from utils.mutation_generators import generate_random_mutations_generator
from utils.visualizations import visualize_score_by_mutation_nr


def investigate_mutational_landscape_monte_carlo(seq, scoring_function, nr_of_samples, max_nr_of_mutations):
    print("Calculating random mutation scores")
    generate_mutations = generate_random_mutations_generator(seq, max_nr_of_mutations, nr_of_samples)
    sequences = []
    nr_of_mutations = []
    for i, mutated_seq in enumerate(generate_mutations()):
        nr_of_mutations_ = np.sum(seq != mutated_seq)
        if nr_of_mutations_ == 0:
            continue
        sequences.append(mutated_seq)
        nr_of_mutations.append(nr_of_mutations_)
    sequences = np.array(sequences)
    results = scoring_function(sequences)
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

    results, nr_of_mutations = investigate_mutational_landscape_monte_carlo(dca.org_seq, dca.score_gpu, 2000, len(dca.org_seq))
    visualize_score_by_mutation_nr(results, nr_of_mutations, "../plots/monte_carlo_sampling.png")
