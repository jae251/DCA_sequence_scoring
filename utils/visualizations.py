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


def visualize_double_mutations(results, path_histogram=None, path_heatmap=None):
    import seaborn as sns
    import matplotlib.pylab as plt
    results = np.array(results)
    print("Loaded data")

    scores = results[:, -2]
    results[:, -1] = scores - scores / results[:, -1]

    g = sns.distplot(results[:, -1], kde=False)
    g.set_yscale("log")
    if path_histogram is None:
        plt.show()
    else:
        plt.savefig(path_histogram)
        print("Saved score difference histogram under ", path_histogram)
    plt.clf()

    dim = 547
    interaction_matrix = np.zeros((dim, dim, 2))
    for p1, r1, p2, r2, score, score_diff in results:
        p1, p2 = int(p1), int(p2)
        pointer = interaction_matrix[p1, p2]
        if pointer[1] > score or pointer[1] == 0:
            interaction_matrix[p1, p2] = score_diff, score
            interaction_matrix[p2, p1] = score_diff, score
    print("Heatmap matrix filled")
    score_differences = interaction_matrix[:, :, 0]
    g = sns.heatmap(score_differences, cmap=sns.diverging_palette(500, 10, sep=1, n=200), center=0)
    g.invert_yaxis()
    if path_heatmap is None:
        plt.show()
    else:
        plt.savefig(path_heatmap)
        print("Saved score difference heatmap under ", path_heatmap)
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
