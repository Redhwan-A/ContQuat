import numpy as np
import scipy as sp
import time
import torch

def normalize_Avec(A_vec):
    """ Normalizes BxM vectors such that resulting symmetric BxNxN matrices have unit Frobenius norm"""
    """ M = N*(N+1)/2"""
    
    A = convert_Avec_to_A(A_vec)
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    A = A / A.norm(dim=[1,2], keepdim=True)
    return convert_A_to_Avec(A).squeeze()

def convert_A_to_Avec(A):
    """ Convert BxNXN symmetric matrices to BxM vectors encoding unique values"""
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    idx = torch.triu_indices(A.shape[1], A.shape[1])
    A_vec = A[:, idx[0], idx[1]]
    return A_vec.squeeze()

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    # print('convert_Avec_to_A: A_vec):', A_vec)
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)
        # print('convert_Avec_to_A: A_vec: unsqueeze):', A_vec)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    # elif A_vec.shape[1] == 55:
    #     A_dim = 10
    # elif A_vec.shape[1] == 6:
    #     A_dim = 3
    # elif A_vec.shape[1] == 15:
    #     A_dim = 5
    # elif A_vec.shape[1] == 21:
    #     A_dim = 6
    #
    # elif A_vec.shape[1] == 28:
    #     A_dim = 7
    #
    # elif A_vec.shape[1] == 36:
    #     A_dim = 8
    # elif A_vec.shape[1] == 45:
    #     A_dim = 9
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")
    # print('convert_Avec_to_A: A_dim):', A_dim)
    idx = torch.triu_indices(A_dim,A_dim)
    # print('convert_Avec_to_A: idx):', idx)
    A = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))
    # print('convert_Avec_to_A: A):', A)
    A[:, idx[0], idx[1]] = A_vec
    # print('convert_Avec_to_A: A[:, idx[0], idx[1]]):', A[:, idx[0], idx[1]])
    A[:, idx[1], idx[0]] = A_vec
    # print('convert_Avec_to_A: A[:, A[:, idx[1], idx[0]]):', A[:, idx[1], idx[0]])
    # print('convert_Avec_to_A: A.squeeze():', A.squeeze())
    return A.squeeze()

def convert_Avec_to_Avec_psd(A_vec):
    """ Convert BxM tensor (encodes symmetric NxN amatrices) to BxM tensor  
    (encodes symmetric and PSD 4x4 matrices)"""

    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze()
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    # elif A_vec.shape[1] == 55:
    #     A_dim = 10
    # elif A_vec.shape[1] == 6:
    #     A_dim = 3
    # elif A_vec.shape[1] == 15:
    #     A_dim = 5
    # elif A_vec.shape[1] == 21:
    #     A_dim = 6
    #
    # elif A_vec.shape[1] == 28:
    #     A_dim = 7
    #
    # elif A_vec.shape[1] == 36:
    #     A_dim = 8
    # elif A_vec.shape[1] == 45:
    #     A_dim = 9
    else:
        raise ValueError("Arbitrary A_vec not yet implementedf")
        print('Arbitrary A_vec not yet implementedf')

    idx = torch.tril_indices(A_dim,A_dim)
    L = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    L[:, idx[0], idx[1]] = A_vec
    A = L.bmm(L.transpose(1,2))
    A_vec_psd = convert_A_to_Avec(A)
    return A_vec_psd



def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)
    # print('A (A_vec):', A, 'A.dim()', A.dim())
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
        # print('A (unsqueeze):', A)
    # _, evs = torch.symeig(A, eigenvectors=True)
    _, evs = torch.linalg.eigh(A) #To convert each 4x4 symmetric matrix into a unit quaternion after finding Eigenvectors
    # print(" Eigenvalues:", _)
    # print("Eigenvectors:\n", evs)
    # print('A (evs):', evs)
    # print('A (evs[:, :, 0].squeeze()):',  evs[:, :, 0].squeeze())
    return evs[:,:,0].squeeze() # 0 to take the first vectors only.


# #=========================PYTORCH (FAST) SOLVER=========================

class QuadQuatFastSolver(torch.autograd.Function):
    """
    Differentiable QCQP solver
    Input: Bx10 tensor 'A_vec' which encodes symmetric 4x4 matrices, A
    Output: q that minimizes q^T A q s.t. |q| = 1
    """

    @staticmethod
    def forward(ctx, A_vec):
        # print('tx, A_vec):',  A_vec)  # [ 3.6498e-01, -7.0026e-02,  7.0684e-02,  3.7599e-02,  6.6994e-01, -1.0776e-01, -1.1622e-01,  6.0266e-01,  2.5622e-04,  2.3965e-02]
        A = convert_Avec_to_A(A_vec)
        # print('A:', A)
        # print('A.size', A.size()) #torch.Size([4, 4])

        if A.dim() < 3:
            # print(' A.dim() ',  A.dim()) #2
            A = A.unsqueeze(dim=0)
            # print('A:, A.dim() ', A, A.dim() ) #3
            # print('A.unsqueeze.size', A.size()) # torch.Size([1, 4, 4])

        q, nu  = solve_wahba_fast(A)
        # print('q, nu  ',q, nu ) #q, nu   tensor([[-0.0765,  0.1688,  0.0388,  0.9819]], device='cuda:0') tensor([[-0.0011]], device='cuda:0')
        # print('q.size() ', q.size()) #torch.Size([1, 4])
        ctx.save_for_backward(A, q, nu)
        # print('ctx  ', ctx)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        A, q, nu = ctx.saved_tensors
        # print('A, q, nu ', A, q, nu )
        grad_qcqp = compute_grad_fast(A, nu, q)
        # print('grad_qcqp  ', grad_qcqp, 'grad_qcqp.size', grad_qcqp.size()) #torch.Size([1, 4, 10])
        outgrad = torch.einsum('bkq,bk->bq', grad_qcqp, grad_output)
        # print('outgrad  ', outgrad, 'outgrad.size', outgrad.size()) #torch.Size([1, 10])
        return outgrad

def solve_wahba_fast(A, compute_gap=False):
    """
    Use a fast eigenvalue solution to the dual of the 'generalized Wahba' problem to solve the primal.
    :param A: quadratic cost matrix
    :param redundant_constraints: boolean indicating whether to use redundand constraints
    :return: Optimal q, optimal dual var. nu, time to solve, duality gap
    """
    #start = time.time()
    # Returns (b,n) and (b,n,n) tensors
    # nus, qs = torch.symeig(A, eigenvectors=True)
    # print('A wahba:', A)
    nus, qs = torch.linalg.eigh(A)
    # print(" Eigenvalues:", nus)
    # print("Eigenvectors:\n", qs)
    # nus = nus[:, 0]  # Extract the real part of the eigenvalues
    nu_min, nu_argmin = torch.min(nus, 1)# , keepdim=False, out=None) nu_argmin= index
    # print("  nu_min, nu_argmin:",  nu_min, nu_argmin)
    q_opt = qs[torch.arange(A.shape[0]), :, nu_argmin] #torch.Size([1, 4])
    # print(" q_opt:", q_opt)
    q_opt = q_opt*(torch.sign(q_opt[:, 3]).unsqueeze(1)) #torch.Size([1, 4])
    # print(" q_opt:", q_opt)
    nu_opt = -1.*nu_min.unsqueeze(1) #torch.Size([1, 1])
    # print(" nu_opt:", nu_opt)
    if compute_gap:
        # print(" compute_gap:", compute_gap)
        p = torch.einsum('bn,bnm,bm->b', q_opt, A, q_opt).unsqueeze(1)
        gap = p + nu_opt
        # print("q_opt, nu_opt, gap:", q_opt, nu_opt, gap)
        return q_opt, nu_opt, gap
    # print("q_opt, nu_opt:", q_opt, nu_opt)
    # print("q_opt.shape, nu_opt.shape:", q_opt.size(), nu_opt.size())
    return q_opt, nu_opt
def compute_grad_fast(A, nu, q):
    """
    Input: A_vec: (B,4,4) tensor (parametrices B symmetric 4x4 matrices)
           nu: (B,) tensor (optimal lagrange multipliers)
           q: (B,4) tensor (optimal unit quaternions)
    Output: grad: (B, 4, 10) tensor (gradient)
    Applies the implicit function theorem to compute gradients of qT*A*q s.t |q| = 1, assuming A is symmetric
    """
    assert(A.dim() > 2 and nu.dim() > 0 and q.dim() > 1)
    M = A.new_zeros((A.shape[0], 5, 5))
    I = A.new_zeros((A.shape[0], 4, 4))
    # print("M, I:", M, I)
    I[:,0,0] = I[:,1,1] = I[:,2,2] = I[:,3,3] = 1.
    # print("I:",  I)
    M[:, :4, :4] = A + I*nu.view(-1,1,1)
    # print("M, I:", M, I)
    M[:, 4,:4] = q
    # print("M", M)
    M[:, :4,4] = q
    # print("M", M)
    b = A.new_zeros((A.shape[0], 5, 10))
    # print("b ", b )
    #symmetric matrix indices
    idx = torch.triu_indices(4,4)
    # print("idx ", idx)
    i = torch.arange(10)
    # print("i ", i)
    I_ij = A.new_zeros((10, 4, 4))
    # print("I_ij ",  I_ij)
    I_ij[i, idx[0], idx[1]] = 1.
    # print(" I_ij ", I_ij)
    I_ij[i, idx[1], idx[0]] = 1.
    # print(" I_ij ", I_ij)
    I_ij = I_ij.expand(A.shape[0], 10, 4, 4)
    # print(" I_ij ", I_ij)
    b[:, :4, :] = torch.einsum('bkij,bi->bjk',I_ij, q)
    # print("b ", b)
    #This solves all gradients simultaneously!
    # X, _ = torch.solve(b, M)
    X = torch.linalg.solve(M,  b)
    # print("X", X)
    grad = -1*X[:,:4,:]
    # print("grad", grad)
    return grad


