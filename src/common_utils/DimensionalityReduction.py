import numpy as np

from src.utils.model_utils.mian_utils import DataDelete

# from model_utils.NewPreDeal_Tools import Del_deletion_data


# =============================================================================
# delete the data with high correlation coefficient (r2_r)
# =============================================================================


if __name__ == '__main__':
    X = np.matrix([[1, 3, 3], [2, 6, 6], [3, 9, 9], [4, 12, 5]])
    ii_l = DataDelete(X, 0.90)
    x_1 = X[:, ii_l]
