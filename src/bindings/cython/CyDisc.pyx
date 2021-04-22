
# %load_ext cython
# %%cython -+ -3 -I/usr/include -I/local/include -I../../../include -I../../../src -ltbb -lgomp -f --compile-args=-std=c++17 --compile-args=-fopenmp
# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "<bindings/cython/CyDisc.hpp>" namespace "sd::disc":
    cdef cppclass tag_sparse
    cdef cppclass tag_dense
    cdef cppclass itemset[T]:
        void insert(int x)
        void clear()
    cdef cppclass Dataset[T]:
        void insert(itemset[T] x)
    dict desc[T](Dataset[T] x, vector[int] y, int min_support, int high_precision_float)
    dict disc[T](Dataset[T] x, int min_support, float alpha, int high_precision_float)

cdef Dataset[tag_dense] to_dataset(xs):
    cdef Dataset[tag_dense] data
    cdef itemset[tag_dense] t
    for x in xs: 
        t.clear()
        for i in x: t.insert(i)
        data.insert(t)
    return data

cdef Dataset[tag_sparse] to_dataset_sparse(xs):
    cdef Dataset[tag_sparse] data
    cdef itemset[tag_sparse] t
    for x in xs: 
        t.clear()
        for i in x: t.insert(i)
        data.insert(t) 
    return data

cdef vector[int] to_vector(x):
    cdef vector[int] y
    y.reserve(len(x))
    for i in x: y.push_back(i)
    return y

class DESC:
    is_sparse = False
    use_high_precision_floats = False
    min_support = 2
    
    initial_objective = (0,0,0)
    objective =  (0,0,0)
    summary = []
    assingment = [] 
    
    def fit(self, D, y=None):
        cdef vector[int] yy
        cdef Dataset[tag_sparse] x_sp #cannot move declaration into if/else branch.
        cdef Dataset[tag_dense] x
        
        labels_given = y is not None and len(y) > 0
        
        if labels_given:
            yy = to_vector(y)
            
        if self.is_sparse:
            x_sp = to_dataset_sparse(D)
            r = desc(x_sp, yy, self.min_support, self.use_high_precision_floats)
        else:
            x = to_dataset(D)
            r = desc(x, yy, self.min_support, self.use_high_precision_floats)
            
        # self.internal_state = r['internal_state']
            self.initial_objective = r['initial_objective']
            self.objective = r['objective']
            self.summary = r['pattern_set']
            if labels_given:
                self.assingment = r['assignment']
        return self.objective[0]
    
    # def predict(D):
    #     if not internal_state: 
    #         raise Exception("Cannot predict without fitting data first")

    #     cdef Dataset[tag_sparse] x_sp 
    #     cdef Dataset[tag_dense] x

    #     if self.is_sparse:
    #         x_sp = to_dataset_sparse(D)
    #         classify(internal_state, x_sp) 
    #     else:
    #         x = to_dataset(D)
    #         r = classify(internal_state, x)
        
    #     return np.arange(r)

    # def shared(is):
    # def characteristic(is):
    # def contrast(is, js):
    # def emerging(is, js):
    # def unique(is):

class DISC:
    is_sparse = False
    use_high_precision_floats = False
    min_support = 2
    alpha = 0.01
    
    initial_objective = (0,0,0)
    objective =  (0,0,0)
    summary = []
    assingment = [] 
    
    def fit(self, D, y=None):
        cdef Dataset[tag_sparse] x_sp #cannot move declaration into if/else branch.
        cdef Dataset[tag_dense] x
            
        if self.is_sparse:
            x_sp = to_dataset_sparse(D)
            r = disc(x_sp, self.min_support, self.alpha, self.use_high_precision_floats)
        else:
            x = to_dataset(D)
            r = disc(x, self.min_support, self.alpha, self.use_high_precision_floats)
            
        self.initial_objective = r['initial_objective']
        self.objective = r['objective']
        self.summary = r['pattern_set']
        self.assingment = r['assignment']
        
        return self.objective[0]
    
de = DESC()
de.fit([[1, 2, 3]])
di = DISC()