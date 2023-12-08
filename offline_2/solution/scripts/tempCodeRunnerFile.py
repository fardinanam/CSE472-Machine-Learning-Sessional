def remainder(X : np.ndarray, y : np.ndarray, feature_index : int) -> float:
#     # find the number of examples
#     n = y.size

#     # find the number of unique values of the feature
#     feature = X[:, feature_index]
#     values = np.unique(feature).flatten()

#     # find the remainder
#     remainder = 0

#     # check if the feature is continuous
#     if len(values) > 10:
#         # find the median of the feature
#         median = np.median(values)
#         # split the feature into two values
#         values_less = values[values <= median].flatten()
#         values_greater = values[values > median].flatten()
#         values = np.concatenate((values_less, values_greater))

#     for v in values:
#         # find the indices of the examples with the value v
#         indices = np.where(feature == v)[0]
#         # find the number of examples with the value v
#         n_v = indices.size
#         # find the entropy of the examples with the value v
#         y_v = y[indices]
#         entropy_v = entropy(y_v)
#         # find the remainder
#         remainder += (n_v / n) * entropy_v

#     return remainder