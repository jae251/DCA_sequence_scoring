import numpy as np


def visualize_score_by_mutation_nr(scores, nr_of_mutations, path=None):
    import matplotlib.pyplot as plt
    colors = create_score_colors(scores)
    plt.axhline()
    plt.scatter(nr_of_mutations, scores, s=5, alpha=.5, facecolors=colors, edgecolors="none")
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
    print("Heatmap matrix filled")
    ax = sns.heatmap(interaction_matrix)  # , linewidth=0.1)
    ax.set_title("Highest synergy scores for mutation pairs")
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


########################################################################################################################

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
