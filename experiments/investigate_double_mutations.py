import numpy as np
import os
from progressbar import ProgressBar
from h5py import File

from utils.mutation_generators import BiasedRandomMutationsGenerator
from investigate_single_mutations import investigate_single_mutations
from utils.visualizations import visualize_double_mutations

cache_file = "../tmp/double_mutations.hdf5"


def investigate_double_mutations(seq, scoring_function, single_mutations_scores):
    print("Calculating double mutations")
    if os.path.isfile(cache_file):
        print("Previous results found in " + cache_file + ", skipping calculation")
    else:
        generate_mutations = BiasedRandomMutationsGenerator(seq, single_mutations_scores,
                                                            mutation_pool_cutoff=len(single_mutations_scores),
                                                            max_nr_of_mutations_per_seq=2)
        with ProgressBar(max_value=generate_mutations.max_nr_of_seq) as bar:
            df = None
            with File(cache_file, "w") as f:
                sequences = []
                mutations = []
                single_scores_added = []
                for i, mutated_seq in enumerate(generate_mutations):
                    sequences.append(mutated_seq)
                    last_mutation = generate_mutations.mutations_of_last_seq
                    last_mutation = last_mutation.reshape(4)
                    mutations.append(last_mutation)
                    single_scores_added.append(single_mutations_scores[(last_mutation[0], last_mutation[1])] +
                                               single_mutations_scores[(last_mutation[2], last_mutation[3])])

                    if i % 65535 == 0 and i != 0:  # using 256 grids with 256 threads on GPU
                        sequences = np.array(sequences)
                        single_scores_added = np.array(single_scores_added)
                        mutations = np.vstack(mutations)
                        scores = scoring_function(sequences)
                        batch_result = np.hstack((mutations, scores[:, np.newaxis], single_scores_added[:, np.newaxis]))
                        if df is None:
                            df = f.create_dataset("results", data=batch_result, maxshape=(None, 6))
                        else:
                            data_size = len(batch_result)
                            df.resize(len(df) + data_size, axis=0)
                            df[-data_size:] = batch_result
                        sequences = []
                        single_scores_added = []
                        mutations = []
                    if i % 10 == 0:
                        bar.update(i)
                sequences = np.array(sequences)
                single_scores_added = np.array(single_scores_added)
                mutations = np.vstack(mutations)
                scores = scoring_function(sequences)
                synergy_coefficient = scores / single_scores_added
                batch_result = np.hstack((mutations, scores[:, np.newaxis], synergy_coefficient[:, np.newaxis]))
                if df is None:
                    f.create_dataset("results", data=batch_result, maxshape=(None, 6))
                else:
                    data_size = len(batch_result)
                    df.resize(len(df) + data_size, axis=0)
                    df[-data_size:] = batch_result
    results_hdf = File(cache_file, "r")
    results = results_hdf["results"]
    return results


if __name__ == '__main__':
    if not os.path.isfile(cache_file):
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
        m2_scores = investigate_double_mutations(dca.org_seq, dca.score_gpu, m1_scores)
    else:
        m2_scores = investigate_double_mutations(None, None, None)
    visualize_double_mutations(m2_scores, "../plots/double_mutations_score_difference_histogram.png",
                               "../plots/double_mutations_score_difference_heatmap.png")
