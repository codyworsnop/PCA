import numpy as np 
import math 

def compute_Z(X, centering=True, scaling=False):

    Z = np.copy(X).astype(float)

    if centering: 

        #calculate the mean of each feature set
        mean = np.mean(Z, axis=0)

        for axis in range(0, Z.shape[1]):   
            for features in Z:
                    features[axis] = features[axis] - mean[axis]

    if scaling: 

        #calculate the standard deviation of the each feature set
        standard_deviation = np.std(Z, axis=0)

        for axis in range(0, Z.shape[1]):
            for features in Z:                
                features[axis] = features[axis] - standard_deviation[axis]

    return Z

def compute_covariance_matrix(Z):
    return np.cov(Z, rowvar=False)

def find_pcs(COV):
    eigen = np.linalg.eig(COV)

    #get the order to sort by
    argsort = np.flip(eigen[0].argsort()) #this reverses order since argsort returns ascending

    #sort the eigenvalues
    sorted_values = eigen[0][argsort]

    #sort the vectors acording to argsort
    sorted_vectors = eigen[1][:,argsort]

    return (sorted_values, sorted_vectors)

def project_data(Z, PCS, L, k, var):

    PCS = np.delete(PCS, range(k, PCS.shape[1]), axis=1)
    z_star = np.transpose(np.dot(np.transpose(PCS), np.transpose(Z)))
    return z_star

def test():
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    Z = compute_Z(X)
    COV = compute_covariance_matrix(Z)
    L, PCS = find_pcs(COV)
    Z_star = project_data(Z, PCS, L, 1, 0)
