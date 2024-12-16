# -*- coding: utf-8 -*-

import argparse
import numpy as np
import scipy.sparse as sp
import torch

import tools as tl

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="sports")

parser.add_argument("-s", "--sampling-size", type=int, default=None)

args = parser.parse_args()
K = args.sampling_size

raw_dir = "../../data/" + args.dataset + "/"

A = torch.load(raw_dir + "adj_0.1.pt")
A = A.coalesce()
A = sp.csr_matrix(
    (
        A.values().cpu().numpy(),
        (A.indices()[0].cpu().numpy(), A.indices()[1].cpu().numpy()),
    ),
    shape=A.size(),
)

if K == 1:
    I = tl.get_mutual_information(A)
    true_sample_index = tl.get_sim(I)
    np.savetxt(raw_dir + "true_sample_index", true_sample_index, fmt="%d")

else:
    if K is not None:
        I = tl.get_mutual_information(A)
        I = I.tocoo()
        I.data = np.exp(I.data)
        I = I.multiply(1 / I.sum(axis = 1))
        I = I.tocsr()
        true_sample_index = tl.get_mutli_sim(I, ratio = 1, K = K)
        
        row = []
        col = []
        i = 0
        for indexes in true_sample_index:
            
            for j in indexes:
                
                row.append(i)
                col.append(j)
                
            i += 1
        
        N = len(row)
           
        mask = sp.coo_matrix((np.ones(N), (row, col)), shape = A.shape).tocsr()
        A = A.multiply(mask)
    
    A = tl.get_L_DA(A)
    
    A = A.tocoo()
    
    rows = A.row
    cols = A.col
    values = A.data
    indices = np.vstack((rows, cols))

    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(values)
    size = torch.Size(A.shape)

    A = torch.sparse_coo_tensor(indices, values, size)

    torch.save(A, raw_dir + f"adj_sampling_0.1_{K}.pt")
