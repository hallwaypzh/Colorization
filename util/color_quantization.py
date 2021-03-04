import numpy as np
import sklearn.neighbors as nn

def flatten_nd_array(pts_nd, axis=1):
    # Flatten an nd array into a 2d array with a certain axis
    # INPUTS
    # 	pts_nd 		N0xN1x...xNd array
    # 	axis 		integer
    # OUTPUTS
    # 	pts_flt 	prod(N \ N_axis) x N_axis array
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    
    #pts_flt = pts_nd.transpose(axorder)
    pts_flt = np.transpose(pts_nd, axorder)
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    # Unflatten a 2d array with a certain axis
    # INPUTS
    # 	pts_flt 	prod(N \ N_axis) x M array
    # 	pts_nd 		N0xN1x...xNd array
    # 	axis 		integer
    # 	squeeze 	bool 	if true, M=1, squeeze it out
    # OUTPUTS
    # 	pts_out 	N0xN1x...xNd array
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def check_value(inds, val):
    # Check to see if an array is a single element equaling a particular value
    # Good for pre-processing inputs in a function
    if(np.array(inds).size == 1):
        if(inds == val):
            return True
    return False


class NNEncode():
    # Encode points as a linear combination of unordered points
    # using NN search and RBF kernel
    def __init__(self, NN, sigma, km_filepath='./util/color_bins/pts_in_hull.npy', cc=-1):
        if(check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='auto').fit(self.cc)

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        (dists, inds) = self.nbrs.kneighbors(pts_flt)
        
        #pts_enc_flt = np.zeros((P, self.K))
        
        pts_enc_flt = np.zeros((P, 529))
        wts = np.exp(-dists**2 / (2 * self.sigma**2))
        wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
        # debug
        # pts_grid = np.load('../data/color_bins/pts_grid.npy')
        inds1 = self.cc[inds]
        #print(inds1.shape)
        inds1 = ((inds1[:, :, 0] / 10 + 11) * 23 + (inds1[:, :, 1]) / 10 + 11).astype('int')
        '''
        print('*' * 100)
        print(pts_flt[0])
        for i in range(1):
            print(self.cc[inds[0]])
            print(wts[0])
            tmp = self.cc[inds[0]]
            print(tmp.shape)
            tmp = ((tmp[:,0] / 10 + 11) * 23 + (tmp[:, 1] / 10 + 11)).astype('int')
            print(pts_grid[tmp])
            #pts_grid[np.int(self.cc[inds[0]][:, 0] / 10 + 11) * 23 + (self.cc[inds[0]][:, 1] / 10 + 11))]
        print('*' * 100)
        '''
        pts_enc_flt[np.arange(0, P, dtype='int')[:, np.newaxis], inds1] = wts
        #pts_enc_flt[np.arange(0, P, dtype='int')[:, util.na()], inds] = wts
        pts_enc_nd = unflatten_2d_array(pts_enc_flt, pts_nd, axis=axis)
        return pts_enc_nd

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return pts_dec_nd
