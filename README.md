This is an implementation of the fair k-median clustering algorithm from the paper:
"Scalable Fair Clustering", by Arturs Backurs, Piotr Indyk, Krzysztof Onak, Baruch Schieber, Ali Vakilian, Tal Wagner (ICML 2019), available at https://arxiv.org/abs/1902.03519

For the definition of fair clustering, please refer to the paper:
"Fair Clustering Through Fairlets", by Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, Sergei Vassilvitskii  (NIPS 2017), available at https://arxiv.org/abs/1902.03519

The main file is fairtree.py. Usage:

-- 1st and 2nd parameters: Two integers that define the desired cluster balance. Namely, for parameters p,q such that p<=q, each output cluster with R red points and B blue points would satisfy min(R/B, B/R) >= p/q.

-- 3rd parameter: number of output clusters (k in k-median)

-- 4th parameter: Dataset, given in CSV format. Each row represents a data point, and is a list of comma-separated numerical values. The first value is 0/1 and represents the sensitive color class (red/blue) of the point, the remaining values are its features.

-- 5th parameter (optional): Integer sample size. If given N, the code would run on a random sample of N records from the dataset. If not given, the code would run on the whole dataset.

The main part of the code computes a fairlet decomposition of the dataset (as defined in Chierichetti et al). Then a usual (non-fair) k-median algorithm is invoked on the fairlet centers. The resulting clustering is then extended to the whole dataset by assigning each data point to the cluster that contains its fairlet center. The yields the final fair clustering.

For usual k-median, the code invokes MATLAB. This requires Python 3.6 or older. Alternatively, the MATLAB invocation can be replaced by any other k-median implementation.
