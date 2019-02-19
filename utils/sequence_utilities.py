import numpy as np
import pandas as pd

RESIDUES = "ACDEFGHIKLMNPQRSTVWY-"
INT_TO_STR = np.array([l for l in RESIDUES])
STR_TO_INT = pd.Series(data=range(len(RESIDUES)), index=INT_TO_STR)


def to_int(seq):
    return STR_TO_INT[seq].values


def to_str(seq):
    return INT_TO_STR[seq]
