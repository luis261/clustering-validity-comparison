import itertools


class LabelSizeMismatchException(Exception):
    pass


def calc_jaccard_coeff(labels0, labels1):
    a = b = c = 0
    if len(labels0) != len(labels1):
        raise LabelSizeMismatchException()

    for i, j in itertools.combinations(range(len(labels0)), 2):
        same_cluster_in_cl0 = labels0[i] == labels0[j]
        same_cluster_in_cl1 = labels1[i] == labels1[j]
        if same_cluster_in_cl0 and same_cluster_in_cl1:
            a += 1
        elif same_cluster_in_cl0 and not same_cluster_in_cl1:
            b += 1
        elif not same_cluster_in_cl0 and same_cluster_in_cl1:
            c += 1

    return a / (a + b + c)