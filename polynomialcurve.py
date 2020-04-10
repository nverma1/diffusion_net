import numpy as np

class PolynomialPointCurve:
    """An object encoding a randomly generated curve and several helper classes to generate noisy data about the curve/check accuracy of points near it"""
    def __init__(self, dimension=3, avg_dims_per_term=2.5, avg_power_per_dim=2, num_terms=6, mono_terms=True, avg_mono_term_power=1, space_size=2, max_coeff=4):
        self.dimension = dimension
        self.space_size = space_size
        self.num_terms = num_terms
        assert (mono_terms == False or num_terms >= dimension)
        self.term_coeffs = np.random.uniform(-max_coeff, max_coeff, num_terms)
        self.powers = np.random.randint(1, avg_power_per_dim*2+1, (num_terms, dimension))
        self.powers[np.random.random((num_terms, dimension))<1-(avg_dims_per_term/dimension)]=0
        if mono_terms:
            self.powers[:dimension,:dimension]=0
            self.powers[np.diag_indices(dimension)] = np.random.randint(1, avg_mono_term_power*2+1, dimension)

        self.threshold = None

    def space_to_unit(data):
        return (data + self.space_size) / (2 * self.space_size)
    def unit_to_space(data):
        return ((data * 2 * self.space_size) - self.space_size)

    def compute_values(self, points): #Computes the polynomial on an input array of tuples. Input array should have dimensions (n_points, self.dimensions)
        points = unit_to_space(points)
        out_vals = np.zeros(points.shape[0])
        for i in range (self.num_terms):
            pow_row = self.powers[i]
            rel_vals = np.where(pow_row>0)
            resultant_pows = np.power(points[:,rel_vals], pow_row[np.newaxis, rel_vals])
            d = np.prod(resultant_pows, axis=(1,2))*self.term_coeffs[i]
            out_vals += d
            
        return np.abs(out_vals)

    def compute_threshold(self, ratio, accuracy_coeff=1000):  #Calculate a nearness-to-0 threshold on the polynomial which encloses <ratio> of the space in the overall space.
        num_pts = int(accuracy_coeff / ratio)
        samples = np.random.uniform(-self.space_size,self.space_size,(num_pts, self.dimension))
        vals = self.compute_values(samples)
        self.threshold = np.sort(vals)[accuracy_coeff]
        return self.threshold

    def compute_error(self, points, mode='mse'):  #Computes the error from 0 of the input points
        points = unit_to_space(points)
        vals = self.compute_values(points)
        num_vals = points.shape[0]
        if mode == 'mse':
            return np.sum(np.square(vals))/num_vals
        elif mode == 'mae':
            return np.sum(vals)/num_vals
        else:
            print("Mode unrecognized! Exiting...")
            exit()

    def gen_noisy_points(self, num_points, error_threshold=None): #Generates points with value around 0 (sampling spatially uniformly across all points below the error threshold)
        if error_threshold is None:
            error_threshold = self.threshold
        good_pts = np.zeros((0,self.dimension))
        BATCH_SIZE = 100000
        while good_pts.shape[0]<num_points:
            prosp_points = np.random.uniform(-self.space_size,self.space_size,(BATCH_SIZE, self.dimension))
            prosp_vals = self.compute_values(prosp_points)
            new_pts = prosp_points[prosp_vals<error_threshold]
            good_pts = np.concatenate([good_pts, new_pts], axis=0)
        return space_to_unit(good_pts[:num_points])



        