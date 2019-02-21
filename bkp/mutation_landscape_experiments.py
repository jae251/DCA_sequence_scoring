import numpy as np
import os
from progressbar import ProgressBar
from h5py import File

from utils.mutation_generators import generate_single_mutations_generator, generate_random_mutations_generator, \
    BiasedRandomMutationsGenerator
from utils.utilities import store_and_load


# @store_and_load
def investigate_double_mutation_synergy(seq, scoring_function, single_mutations_scores):
    print("Calculating double mutation synergy")
    generate_mutations = BiasedRandomMutationsGenerator(seq, single_mutations_scores,
                                                        mutation_pool_cutoff=len(single_mutations_scores),
                                                        max_nr_of_mutations_per_seq=2)
    # results = []
    with ProgressBar(max_value=generate_mutations.max_nr_of_seq) as bar:
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

            if i % 65536 == 0 and i != 0:  # using 256 grids with 256 threads on GPU
                sequences = np.array(sequences)
                single_scores_added = np.array(single_scores_added)
                mutations = np.vstack(mutations)
                scores = scoring_function(sequences)
                synergy_coefficient = scores / single_scores_added
                batch_result = np.hstack((mutations, scores[:, np.newaxis], synergy_coefficient[:, np.newaxis]))
                with File("tmp/double_mutations/{}.hdf5".format(i)) as f:
                    f.create_dataset("results", data=batch_result)
                # results.append(batch_result)
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
        with File("tmp/double_mutations/{}.hdf5".format(i)) as f:
            f.create_dataset("results", data=batch_result)
        # results.append(batch_result)
        # results = np.vstack(results)
    # return results
    return "tmp/double_mutations"


def investigate_mutational_landscape_biased(seq, scoring_function, single_mutations_scores,
                                            mutation_pool_cutoff=None,
                                            max_nr_of_mutations_per_seq=None,
                                            max_nr_of_seq=None):
    print("Calculating biased combination trial")
    generate_mutations = BiasedRandomMutationsGenerator(seq, single_mutations_scores, mutation_pool_cutoff,
                                                        max_nr_of_mutations_per_seq, max_nr_of_seq)
    results = []
    with ProgressBar(max_value=generate_mutations.max_nr_of_seq) as bar:
        for i, mutated_seq in enumerate(generate_mutations):
            nr_of_mutations = np.sum(seq != mutated_seq)
            # results.append(nr_of_mutations)
            score = scoring_function(mutated_seq)
            # print(i, score)
            if i % 10 == 0:
                bar.update(i)
            results.append((score, nr_of_mutations))
    # print(max(results))
    results = np.array(results, dtype=np.dtype([("dca_score", np.float32), ("nr_of_mutations", np.uint16)]))
    single_mutations_results = np.ones(len(generate_mutations.single_mutation_scores),
                                       dtype=np.dtype([("dca_score", np.float32), ("nr_of_mutations", np.uint16)]))
    single_mutations_results["dca_score"] = generate_mutations.single_mutation_scores
    results = np.append(results, single_mutations_results)
    return results


def investigate_mutational_landscape_monte_carlo(seq, scoring_function, nr_of_samples, max_nr_of_mutations):
    print("Calculating random mutation scores")
    generate_mutations = generate_random_mutations_generator(seq, max_nr_of_mutations, nr_of_samples)
    results = np.zeros(nr_of_samples, dtype=np.dtype([("dca_score", np.float32), ("nr_of_mutations", np.uint16)]))
    with ProgressBar(max_value=generate_mutations.max_nr_of_seq) as bar:
        for i, mutated_seq in enumerate(generate_mutations()):
            nr_of_mutations = np.sum(seq != mutated_seq)
            if nr_of_mutations == 0:
                continue
            score = scoring_function(mutated_seq)
            # print(i, nr_of_mutations, score)
            if i % 10 == 0:
                bar.update(i)
            results[i] = score, nr_of_mutations
    return results


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


########################################################################################################################
def visualize(results, path=None):
    import matplotlib.pyplot as plt
    scores = results["dca_score"]
    colors = create_score_colors(scores)
    plt.axhline()
    plt.scatter(results["nr_of_mutations"], scores, s=5, alpha=.5, facecolors=colors, edgecolors="none")
    plt.xlabel("Number of mutations")
    plt.ylabel("DCA score")
    plt.gca().invert_yaxis()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
        print("Saved mutations plot under ", path)
    plt.clf()


def visualize_mutation_synergy(results, path=None):
    import seaborn as sns
    import matplotlib.pylab as plt
    # max_pos = int(max(np.max(results[:, 0]), np.max(results[:, 2])))
    # max_res = int(max(np.max(results[:, 1]), np.max(results[:, 3])))
    # dim = max_pos * max_res + 1
    dim = 547
    interaction_matrix = np.zeros((dim, dim))
    for p1, r1, p2, r2, _, c in results:
        p1, p2 = int(p1), int(p2)
        # interaction_matrix[int(p1) * max_res + int(r1), int(p2) * max_res + int(r2)] = c
        pointer = interaction_matrix[p1, p2]
        if pointer < c or pointer == 0:
            interaction_matrix[p1, p2] = c
    print("Matrix filled")
    ax = sns.heatmap(interaction_matrix)  # , linewidth=0.1)
    ax.set_title("Synergy scores for mutation pairs")
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        print("Saved pairwise mutation synergy plot under ", path)
    plt.clf()


def visualize_single_mutations(scores, path=None):
    import matplotlib.pyplot as plt
    mut_pos = [m[0] for m in scores.keys()]
    colors = create_score_colors(scores.values())
    plt.axhline()
    plt.scatter(mut_pos, scores.values(), s=5, alpha=.5, edgecolors="none", facecolors=colors, marker=None)
    plt.gca().invert_yaxis()
    plt.xlabel("Sequence position number")
    plt.ylabel("DCA score")
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
        print("Saved single mutation plot under ", path)
    plt.clf()


def create_score_colors(scores):
    '''
    Map scores to color gradient, green for negative, red for positive scores, black around 0
    '''
    colors = np.zeros((len(scores), 3))
    for i, score in enumerate(scores):
        if score < 0:
            colors[i] = 0, score, 0
        elif score > 0:
            colors[i] = score, 0, 0
        else:
            colors[i] = 0, 0, 0
    colors[colors[:, 1] < 0] /= colors.min()
    colors[colors[:, 0] > 0] /= colors.max()
    return colors


########################################################################################################################

def usecase_single_mutations(dca):
    m1_scores = investigate_single_mutations(dca.org_seq, dca.score_gpu)
    visualize_single_mutations(m1_scores, os.path.join("plots", "single_mutations.png"))


def usecase_monte_carlo_sampling(dca):
    results = investigate_mutational_landscape_monte_carlo(dca.org_seq, dca.score_gpu, 1000, len(dca.org_seq))
    visualize(results, os.path.join("plots", "monte_carlo_sampling.png"))


def usecase_combinations_of_best_single_mutations(dca):
    m1_scores = investigate_single_mutations(dca.org_seq, dca.score_gpu)
    results = investigate_mutational_landscape_biased(dca.org_seq, dca.score_gpu, m1_scores)
    visualize(results, os.path.join("plots", "best_mutations.png"))


def usecase_double_mutation_synergy(dca):
    m1_scores = investigate_single_mutations(dca.org_seq, dca.score_gpu)
    results = investigate_double_mutation_synergy(dca.org_seq, dca.score_gpu, m1_scores)
    visualize_mutation_synergy(results, path=os.path.join("plots", "double_mutation_synergy.png"))


if __name__ == '__main__':
    from dca import DCAScoreGen
    from time import time
    import pickle

    t1 = time()
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    if os.path.isfile(os.path.join("tmp", "dca.pickle")):
        print("Loading DCA model from file")
        with open(os.path.join("tmp", "dca.pickle"), "rb") as f:
            dca = pickle.load(f)
    else:
        print("Calculating DCA model")
        dca = DCAScoreGen("sample_data/9C542562-5C10-11E7-AF2C-EEEADBC3747A.3.afa")
        with open(os.path.join("tmp", "dca.pickle"), "wb") as f:
            pickle.dump(dca, f)
    t2 = time()

    ##################################################

    # usecase_single_mutations(dca)
    # usecase_monte_carlo_sampling(dca)
    # usecase_combinations_of_best_single_mutations(dca)
    usecase_double_mutation_synergy(dca)
    ##################################################

    t3 = time()
    print(t2 - t1)
    print(t3 - t2)
