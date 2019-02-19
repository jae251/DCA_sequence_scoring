import numpy as np
from utils.sequence_utilities import RESIDUES
from itertools import combinations

from utils.utilities import count_combinations


def generate_single_mutations_generator(seq):
    nr_of_res = len(RESIDUES)

    def single_mutations_generator():
        for pos, local_res in enumerate(seq):
            for res in range(nr_of_res):
                if local_res == res:
                    continue
                yield pos, res

    return single_mutations_generator


def generate_random_mutations_generator(seq, max_nr_of_mutations, samples):
    residues = np.arange(len(RESIDUES))
    seq_idx = np.arange(len(seq))

    def generate_mutations_randomly():
        for _ in range(samples):
            nr_of_mutations = np.random.randint(0, max_nr_of_mutations)
            pos = np.random.choice(seq_idx, nr_of_mutations, replace=False)
            mutations = np.random.choice(residues, nr_of_mutations)
            # randomly drawn mutation can be equal to unmutated residue -> actual nr_of_mutations are <= nr_of_mutations
            # => necessary to check nr of mutations afterwards: nr = np.sum(seq == mut_seq)
            mutated_seq = seq.copy()
            mutated_seq[pos] = mutations
            yield mutated_seq

    return generate_mutations_randomly


class BiasedRandomMutationsGenerator:
    '''
    Uses single mutation scores to generate mutation combination from the best single mutations.
    :param seq: The sequence to mutate as array of ints
    :param single_mutations_scores: List of scores of mutations ordered the same way that single_mutations_generator
            generates them.
    :param mutation_pool_cutoff: how many best single mutations to use. Defaults to all mutations with negative score.
    :param max_nr_of_mutations_per_seq: Maximum number of mutations to original sequence. Defaults to len(seq).
    :param max_nr_of_seq: Limit of generated sequences. Defaults to previous params determining limit.
    '''

    def __init__(self, seq, single_mutations_scores,
                 mutation_pool_cutoff=None, max_nr_of_mutations_per_seq=None, max_nr_of_seq=None):
        self.seq = seq
        self.iteration = 0
        self.mutation_counter = 2
        if max_nr_of_mutations_per_seq is None:
            self.max_nr_of_mutations_per_seq = len(seq)
        else:
            self.max_nr_of_mutations_per_seq = max_nr_of_mutations_per_seq
        sorted_scores, sorted_indices = zip(*sorted(zip(single_mutations_scores.values(),
                                                        single_mutations_scores.keys())))
        if mutation_pool_cutoff is None:
            print("Selecting all beneficial single mutations")
            for i, score in enumerate(sorted_scores):
                if score >= 0:
                    self.mutation_pool_cutoff = i
                    break
        else:
            self.mutation_pool_cutoff = mutation_pool_cutoff

        self.sorted_mutations = sorted_indices[:self.mutation_pool_cutoff]

        print("Selected {} single mutations".format(self.mutation_pool_cutoff), end=" ")
        mutation_positions = np.array(self.sorted_mutations)[:, 0]
        nr_of_unique_mutated_positions = len(np.unique(mutation_positions))
        print("on {} unique sequence positions".format(nr_of_unique_mutated_positions))

        print("Maximum mutations per sequence: {}".format(self.max_nr_of_mutations_per_seq))

        nr_of_combinations = count_combinations(mutation_positions,
                                                self.max_nr_of_mutations_per_seq)
        print("The selected options amount to {} possible mutation combinations ".format(nr_of_combinations))
        if max_nr_of_seq is None:
            self.max_nr_of_seq = nr_of_combinations
        else:
            self.max_nr_of_seq = max_nr_of_seq
            print("Will stop at {} calculated sequences".format(self.max_nr_of_seq))

        self.single_mutation_scores = sorted_scores[:self.mutation_pool_cutoff]
        self.mutation_combinations = combinations(self.sorted_mutations, self.mutation_counter)
        self.mutations_of_last_seq = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.iteration >= self.max_nr_of_seq:
                raise StopIteration
            try:
                mutations = next(self.mutation_combinations)
            except StopIteration:
                self.mutation_counter += 1
                if self.mutation_counter > self.mutation_pool_cutoff:
                    raise StopIteration
                if self.mutation_counter > self.max_nr_of_mutations_per_seq:
                    raise StopIteration
                self.mutation_combinations = combinations(self.sorted_mutations, self.mutation_counter)
                mutations = next(self.mutation_combinations)
            mutations = np.array(mutations)
            if len(mutations) == len(np.unique(mutations[:, 0])):
                self.mutations_of_last_seq = mutations
                mutated_seq = self._mutate_seq(mutations[:, 0], mutations[:, 1])
                self.iteration += 1
                break
        return mutated_seq

    def _mutate_seq(self, positions, residues):
        seq = self.seq.copy()
        seq[positions] = residues
        return seq

    def _mut_idx_to_mut_instruction(self, n):
        mutations_per_positions = len(RESIDUES) - 1
        pos = int(n / mutations_per_positions)
        residue = n - pos * mutations_per_positions
        org_residue = self.seq[pos]
        if org_residue <= residue:
            residue += 1
        return pos, residue
