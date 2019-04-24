### The MATLAB module does not work with python 3.7.0. Use 3.6.0 release or older.
 
import sys
import numpy as np
from collections import defaultdict
import matlab.engine
import time
 
 
EPSILON = 0.0001
 
FAIRLETS = []
FAIRLET_CENTERS = []
 
class TreeNode:
 
    def __init__(self):
        self.children = []
 
    def set_cluster(self, cluster):
        self.cluster = cluster
 
    def add_child(self, child):
        self.children.append(child)
 
    def populate_colors(self, colors):
        "Populate auxiliary lists of red and blue points for each node, bottom-up"
        self.reds = []
        self.blues = []
        if len(self.children) == 0:
            # Leaf
            for i in self.cluster:
                if colors[i] == 0:
                    self.reds.append(i)
                else:
                    self.blues.append(i)
        else:
            # Not a leaf
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)
 
 
### K-MEDIAN CODE ###
 
def kmedian_cost(points, centroids, dataset):
    "Computes and returns k-median cost for given dataset and centroids"
    return sum(np.amin(np.concatenate([np.linalg.norm(dataset[:,:]-dataset[centroid,:], axis=1).reshape((dataset.shape[0], 1)) for centroid in centroids], axis=1), axis=1))

def fair_kmedian_cost(centroids, dataset):
    "Return the fair k-median cost for given centroids and fairlet decomposition"
    total_cost = 0
    for i in range(len(FAIRLETS)):
        # Choose index of centroid which is closest to the i-th fairlet center
        cost_list = [np.linalg.norm(dataset[centroids[j],:]-dataset[FAIRLET_CENTERS[i],:]) for j in range(len(centroids))]
        cost, j = min((cost, j) for (j, cost) in enumerate(cost_list))
        # Assign all points in i-th fairlet to above centroid and compute cost
        total_cost += sum([np.linalg.norm(dataset[centroids[j],:]-dataset[point,:]) for point in FAIRLETS[i]])
    return total_cost
 
 
### FAIRLET DECOMPOSITION CODE ###
 
def balanced(p, q, r, b):
    if r==0 and b==0:
        return True
    if r==0 or b==0:
        return False
    return min(r*1./b, b*1./r) >= p*1./q
 
 
def make_fairlet(points, dataset):
    "Adds fairlet to fairlet decomposition, returns median cost"
    FAIRLETS.append(points)
    cost_list = [sum([np.linalg.norm(dataset[center,:]-dataset[point,:]) for point in points]) for center in points]
    cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
    FAIRLET_CENTERS.append(points[center])
    return cost
 
 
def basic_fairlet_decomposition(p, q, blues, reds, dataset):
    """
    Computes vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).
    Returns cost.
    Input: Balance parameters p,q which are non-negative integers satisfying p<=q and gcd(p,q)=1.
    "blues" and "reds" are sets of points indices with balance at least p/q.
    """
    assert p <= q, "Please use balance parameters in the correct order"
    if len(reds) < len(blues):
        temp = blues
        blues = reds
        reds = temp
    R = len(reds)
    B = len(blues)
    assert balanced(p, q, R, B), "Input sets are unbalanced: "+str(R)+","+str(B)
 
    if R==0 and B==0:
        return 0
 
    b0 = 0
    r0 = 0
    cost = 0
    while (R-r0)-(B-b0) >= q-p and R-r0 >= q and B-b0 >= p:
        cost += make_fairlet(reds[r0:r0+q]+blues[b0:b0+p], dataset)
        r0 += q
        b0 += p
    if R-r0 + B-b0 >=1 and R-r0 + B-b0 <= p+q:
        cost += make_fairlet(reds[r0:]+blues[b0:], dataset)
        r0 = R
        b0 = B
    elif R-r0 != B-b0 and B-b0 >= p:
        cost += make_fairlet(reds[r0:r0+(R-r0)-(B-b0)+p]+blues[b0:b0+p], dataset)
        r0 += (R-r0)-(B-b0)+p
        b0 += p
    assert R-r0 == B-b0, "Error in computing fairlet decomposition"
    for i in range(R-r0):
        cost += make_fairlet([reds[r0+i], blues[b0+i]], dataset)
    return cost
 
 
def node_fairlet_decomposition(p, q, node, dataset, donelist, depth):

    # Leaf                                                                                          
    if len(node.children) == 0:
        node.reds = [i for i in node.reds if donelist[i]==0]
        node.blues = [i for i in node.blues if donelist[i]==0]
        assert balanced(p, q, len(node.reds), len(node.blues)), "Reached unbalanced leaf"
        return basic_fairlet_decomposition(p, q, node.blues, node.reds, dataset)
 
    # Preprocess children nodes to get rid of points that have already been clustered
    for child in node.children:
        child.reds = [i for i in child.reds if donelist[i]==0]
        child.blues = [i for i in child.blues if donelist[i]==0]
 
    R = [len(child.reds) for child in node.children]
    B = [len(child.blues) for child in node.children]
 
    if sum(R) == 0 or sum(B) == 0:
        assert sum(R)==0 and sum(B)==0, "One color class became empty for this node while the other did not"
        return 0
 
    NR = 0
    NB = 0
 
    # Phase 1: Add must-remove nodes
    for i in range(len(node.children)):
        if R[i] >= B[i]:
            must_remove_red = max(0, R[i] - int(np.floor(B[i]*q*1./p)))
            R[i] -= must_remove_red
            NR += must_remove_red
        else:
            must_remove_blue = max(0, B[i] - int(np.floor(R[i]*q*1./p)))
            B[i] -= must_remove_blue
            NB += must_remove_blue
 
    # Calculate how many points need to be added to smaller class until balance
    if NR >= NB:
        # Number of missing blues in (NR,NB)
        missing = max(0, int(np.ceil(NR*p*1./q)) - NB)
    else:
        # Number of missing reds in (NR,NB)
        missing = max(0, int(np.ceil(NB*p*1./q)) - NR)
         
    # Phase 2: Add may-remove nodes until (NR,NB) is balanced or until no more such nodes
    for i in range(len(node.children)):
        if missing == 0:
            assert balanced(p, q, NR, NB), "Something went wrong"
            break
        if NR >= NB:
            may_remove_blue = B[i] - int(np.ceil(R[i]*p*1./q))
            remove_blue = min(may_remove_blue, missing)
            B[i] -= remove_blue
            NB += remove_blue
            missing -= remove_blue
        else:
            may_remove_red = R[i] - int(np.ceil(B[i]*p*1./q))
            remove_red = min(may_remove_red, missing)
            R[i] -= remove_red
            NR += remove_red
            missing -= remove_red
 
    # Phase 3: Add unsatuated fairlets until (NR,NB) is balanced
    for i in range(len(node.children)):
        if balanced(p, q, NR, NB):
            break
        if R[i] >= B[i]:
            num_saturated_fairlets = int(R[i]/q)
            excess_red = R[i] - q*num_saturated_fairlets
            excess_blue = B[i] - p*num_saturated_fairlets
        else:
            num_saturated_fairlets = int(B[i]/q)
            excess_red = R[i] - p*num_saturated_fairlets
            excess_blue = B[i] - q*num_saturated_fairlets
        R[i] -= excess_red
        NR += excess_red
        B[i] -= excess_blue
        NB += excess_blue
 
    assert balanced(p, q, NR, NB), "Constructed node sets are unbalanced"
 
    reds = []
    blues = []
    for i in range(len(node.children)):
        for j in node.children[i].reds[R[i]:]:
            reds.append(j)
            donelist[j] = 1
        for j in node.children[i].blues[B[i]:]:
            blues.append(j)
            donelist[j] = 1
 
    assert len(reds)==NR and len(blues)==NB, "Something went horribly wrong"
 
    return basic_fairlet_decomposition(p, q, blues, reds, dataset) + sum([node_fairlet_decomposition(p, q, child, dataset, donelist, depth+1) for child in node.children])
 
 
def tree_fairlet_decomposition(p, q, root, dataset, colors):
    "Main fairlet clustering function, returns cost wrt original metric (not tree metric)"
    assert p <= q, "Please use balance parameters in the correct order"
    root.populate_colors(colors)
    assert balanced(p, q, len(root.reds), len(root.blues)), "Dataset is unbalanced"
    root.populate_colors(colors)
    donelist = [0] * dataset.shape[0]
    return node_fairlet_decomposition(p, q, root, dataset, donelist, 0)
 
 
### QUADTREE CODE ###
 
def build_quadtree(dataset, max_levels=0, random_shift=True):
    "If max_levels=0 there no level limit, quadtree will partition until all clusters are singletons"
    dimension = dataset.shape[1]
    lower = np.amin(dataset, axis=0)
    upper = np.amax(dataset, axis=0)
 
    shift = np.zeros(dimension)
    if random_shift:
        for d in range(dimension):
            spread = upper[d] - lower[d]
            shift[d] = np.random.uniform(0, spread)
            upper[d] += spread
 
    return build_quadtree_aux(dataset, range(dataset.shape[0]), lower, upper, max_levels, shift)
     
 
def build_quadtree_aux(dataset, cluster, lower, upper, max_levels, shift):
    """
    "lower" is the "bottom-left" (in all dimensions) corner of current hypercube
    "upper" is the "upper-right" (in all dimensions) corner of current hypercube
    """
 
    dimension = dataset.shape[1]
    cell_too_small = True
    for i in range(dimension):
        if upper[i]-lower[i] > EPSILON:
            cell_too_small = False
 
    node = TreeNode()
    if max_levels==1 or len(cluster)<=1 or cell_too_small:
        # Leaf
        node.set_cluster(cluster)
        return node
     
    # Non-leaf
    midpoint = 0.5 * (lower + upper)
    subclusters = defaultdict(list)
    for i in cluster:
        subclusters[tuple([dataset[i,d]+shift[d]<=midpoint[d] for d in range(dimension)])].append(i)
    for edge, subcluster in subclusters.items():
        sub_lower = np.zeros(dimension)
        sub_upper = np.zeros(dimension)
        for d in range(dimension):
            if edge[d]:
                sub_lower[d] = lower[d]
                sub_upper[d] = midpoint[d]
            else:
                sub_lower[d] = midpoint[d]
                sub_upper[d] = upper[d]
        node.add_child(build_quadtree_aux(dataset, subcluster, sub_lower, sub_upper, max_levels-1, shift))
    return node
 
 
### MAIN ###
 
if len(sys.argv) < 4:
    print("Usage:")
    print("First and second parameters are non-negative coprime integers that specify the target balance")
    print("Third parameter is k for k-clustering")
    print("Fourth parameters is a file in CSV format, where each line is a data point, the first column is 0/1 specifying the colors, and the rest of the columns are numerical specifying the point coordinates.")
    print("Fifth parameter is optional, non-negative integer to determine sample size. If given, that number of points will be randomly sampled from the dataset. If not given, will run on the whole dataset.")
    print("Terminating")
    sys.exit(0)
 
try:
    p = min(int(sys.argv[1]), int(sys.argv[2]))
    q = max(int(sys.argv[1]), int(sys.argv[2]))
except:
    print("First two parameters must be non-negative integers that specify the target balance; terminating")
    sys.exit(0)
 
k = int(sys.argv[3])
 
if len(sys.argv) > 4:
    # Parse input file in CSV format, first column is colors, other columns are coordinates
    print("Loading data from input CSV file")
    input_csv_filename = sys.argv[4]
    colors = []
    points = []
    i = 0
    skipped_lines = 0
    for line in open(input_csv_filename).readlines():
        if len(line.strip()) == 0:
            skipped_lines += 1
            continue
        tokens = line[:-1].split(",")
        try:
            color = int(tokens[0])
        except:
            print("Invalid color label in line", i, ", skipping")
            skipped_lines += 1
            continue
        try:
            point = [float(x) for x in tokens[1:]]
        except:
            print("Invalid point coordinates in line", i, ", skipping")
            skipped_lines += 1
            continue
        colors.append(color)
        points.append(point)
        i += 1
 
    n_points = len(points)
    if  n_points == 0:
        print("No successfully parsed points in input file, terminating")
        sys.exit(0)
    dimension = len(points[0])
 
    dataset = np.zeros((n_points, dimension))
    for i in range(n_points):
        if len(points[i]) < dimension:
            print("Insufficient dimension in line", i+skipped_lines, ", terminating")
            sys.exit(0)
        for j in range(dimension):
            dataset[i,j] = points[i][j]
 
else:
    print("No input file given; randomizing data")
    n_points = 1000
    dimension = 5
    dataset = np.random.random((n_points, dimension))
    colors = [np.random.randint(2) for i in range(n_points)]
 
if len(sys.argv) > 5:
    sample_size = int(sys.argv[5])
    idx = np.arange(n_points)
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    n_points = sample_size
    dataset = dataset[idx,:]
    colors = [colors[i] for i in idx]
     
print("Number of data points:", n_points)
print("Dimension:", dimension)
print("Balance:", p, q)
 
print("Constructing tree...")
fairlet_s = time.time()
root = build_quadtree(dataset)
 
print("Doing fair clustering...")
cost = tree_fairlet_decomposition(p, q, root, dataset, colors)
fairlet_e = time.time()
 
#print("Fairlet decomposition cost:", cost)
 
print("Doing k-median clustering on fairlet centers...")
###centroids = kmedian_swap(k, FAIRLET_CENTERS, dataset)
 
 
 
fairlet_center_idx = [dataset[index] for index in FAIRLET_CENTERS]
fairlet_center_pt = np.array([np.array(xi) for xi in fairlet_center_idx])
 
# convert points into matlab array format
mat_matrix = matlab.double(fairlet_center_pt.tolist())
 
# Run k-mediod code in Matlab
cluster_s = time.time()
eng = matlab.engine.start_matlab()
 
# C: Cluster medoid locations, returned as a numeric matrix.
# C is a k-by-d matrix, where row j is the medoid of cluster j
#
# midx: Index to mat_matrix, returned as a column vector of indices.
# midx is a k-by-1 vector and the indices satisfy C = X(midx,:)
idx,C,sumd,D,midx,info = eng.kmedoids(mat_matrix, k,'Distance','euclidean', nargout=6)
cluster_e = time.time()
 
#np_idx = (np.array(idx._data)).flatten()
 
# compute the indices of centers returned by Matlab in its input matrix
# which is mat_matrix or fairlet_center_pt
np_midx = (np.array(midx._data)).flatten()
c_idx_matrix = np_midx.astype(int)
#in matlab, arrays are numbered from 1
c_idx_matrix[:] = [index - 1 for index in c_idx_matrix]
 
# indices of center points in dataset
centroids = [FAIRLET_CENTERS[index] for index in c_idx_matrix]
 
print("Computing fair k-median cost...")
kmedian_cost = fair_kmedian_cost(centroids, dataset)
print("Fairlet decomposition cost:", cost)
print("k-Median cost:", kmedian_cost)
