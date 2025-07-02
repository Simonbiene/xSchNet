import numpy as np
import math
import scipy
from scipy.special import sph_harm, factorial, binom

from pyscf import gto, scf
from pyscf.lo import PM

import os
import csv
import matplotlib.pyplot as plt
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
from schnetpack.nn.activations import shifted_softplus
from atoms import *
import torch
from torch import nn
from torch.utils.data import Dataset
from ase import Atoms
from ase.db import connect
from time import time
from itertools import permutations


n_features = 128
a0 = 0.5291772 # Angstrom
Hartree_to_eV = 27.211386256
Hartree_to_kcal_per_mol = 0.04338
atom_symbols = ['H','He','Li','Na','B','C','N','O']
atom_names = ["hydrogen",None,None,None,None,"carbon","nitrogen","oxygen"]
atomrefs = [0,-13.6131,0,0,0,0,-1029.863,-1485.3025,-2042.6111,-2713.4849]
spin = [1,0,1,0,1,2,3,2,1,0]
mean = -4.2427 # eV


class Atomwise(torch.nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        activation,
        n_in: int,
        n_out: int = 1,
        n_hidden = None,
        n_layers: int = 2,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key = None,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        inputs[self.output_key] = y
        return inputs


class extract_output_keys(torch.nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(self,
                 output_key="scalar_representation",
                 per_atom_output_key = None):
        super(extract_output_keys, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return inputs


class ShiftedSoftplus(torch.nn.Module):
    """
    Shifted softplus activation function with learnable feature-wise parameters:
    f(x) = alpha/beta * (softplus(beta*x) - log(2))
    softplus(x) = log(exp(x) + 1)
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    With learnable parameters alpha and beta, the shifted softplus function can
    become equivalent to ReLU (if alpha is equal 1 and beta approaches infinity) or to
    the identity function (if alpha is equal 2 and beta is equal 0).
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            initial_alpha: Initial "scale" alpha of the softplus function.
            initial_beta: Initial "temperature" beta of the softplus function.
            trainable: If True, alpha and beta are trained during optimization.
        """
        super(ShiftedSoftplus, self).__init__()
        initial_alpha = torch.tensor(initial_alpha)
        initial_beta = torch.tensor(initial_beta)

        if trainable:
            self.alpha = torch.nn.Parameter(torch.FloatTensor([initial_alpha]))
            self.beta = torch.nn.Parameter(torch.FloatTensor([initial_beta]))
        else:
            self.register_buffer("alpha", initial_alpha)
            self.register_buffer("beta", initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Args:
            x (FloatTensor [:, num_features]): Input features.

        Returns:
            y (FloatTensor [:, num_features]): Activated features.
        """
        return self.alpha * torch.where(
            self.beta != 0,
            (torch.nn.functional.softplus(self.beta * x) - math.log(2)) / self.beta,
            0.5 * x,
        )


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation = spk.nn.shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = spk.nn.Dense(n_atom_basis, n_filters, bias=True, activation=None)
        self.f2out = nn.Sequential(
            spk.nn.Dense(n_filters, n_atom_basis, activation=activation),
            spk.nn.Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            spk.nn.Dense(n_rbf, n_filters, activation=activation), spk.nn.Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = spk.nn.scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x

class SchNOrbInteraction(nn.Module):

    def __init__(self,n_factors,n_rbf,idx_j,M):
        super(SchNOrbInteraction, self).__init__()

        self.idx_j = idx_j
        self.M = M
        self.filter_network = nn.Sequential(
            spk.nn.Dense(n_rbf, n_factors, activation=spk.nn.shifted_softplus),
            spk.nn.Dense(n_factors, n_factors)
        )
        self.pairnet = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors, activation=spk.nn.shifted_softplus),
            spk.nn.Dense(n_factors, n_factors)
        )
        self.envnet = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors, activation=spk.nn.shifted_softplus),
            spk.nn.Dense(n_factors, n_factors)
        )

        self.f2out = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors, activation=spk.nn.shifted_softplus),
            spk.nn.Dense(n_factors, n_factors)
        )

        self.n_factors = n_factors


    def forward(self,x,f_ij):
        batch_size = x.shape[0]

        W = self.filter_network(f_ij)
        xj = torch.gather(x,1,self.idx_j[None,:,None].expand(batch_size,-1,self.n_factors)).reshape((batch_size,self.M,self.M-1,self.n_factors))
        h_ij = torch.einsum('nki,nkli->nkli', x, torch.einsum('nkli,nkli->nkli',W,xj) )

        p_ij_pair = self.pairnet(h_ij)
        p_ij_env = self.envnet(h_ij)
        p_ij = p_ij_pair + torch.sum(p_ij_env,dim=1)[:,None,:,:] + torch.sum(p_ij_env,dim=2)[:,:,None,:]
    
        v = self.f2out(torch.sum(h_ij,dim=2))

        return p_ij, v


class SchNet_XAIDataSet(torch.utils.data.Dataset):
    def __init__(self, targets, target_E, inputs):
        # create input and target data
        self.input_data = inputs
        self.target_E = target_E
        
        F0 = targets[0,:,:,:]
        h0 = targets[1,:,:,:]
        S0 = targets[2,:,:,:]
        
        # transform in orthogonal basis
        s, L = torch.linalg.eigh(S0)
        self.S_inv_root = L @ torch.diag_embed(torch.sqrt(1/s)) @ L.transpose(1,2)
        self.h_new = torch.einsum('npr,nrq->npq',self.S_inv_root,torch.einsum('nrs,nsq->nrq', h0, self.S_inv_root))
        
        #self.grad_h_new = torch.zeros(self.h_new.shape)
        #self.grad_h_new[:-1] = (self.h_new[1:] - self.h_new[:-1])/(hi-lo)*increments


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, n):
        input = (self.input_data[n], self.h_new[n,:,:]) # self.S_inv_root[n,:,:], 
        target = self.target_E[n] # ,self.h_new[n,:,:],self.J_K_new[n,:,:])
        return input, target


class XAI_model_linear(nn.Module):
    def __init__(self,formula,basis,F_dissoc,idx_i,idx_j,numbers):
        super(XAI_model_linear, self).__init__()
        
        B, B_k, N, M = get_basis_size(basis,numbers)

        self.formula = formula
        self.numbers = numbers
        self.B = B
        self.n_atoms = M
        self.idx_i = idx_i
        self.idx_j = idx_j
        mask = get_mask(basis,B)
        self.N = N

        if formula=='N2':
            self.register_buffer('a', 0.7*torch.ones(1,dtype=torch.double))
            self.register_buffer('b', F_dissoc)
            self.register_buffer('mask',mask)
            A_on = (torch.rand(B//2,B//2,dtype=torch.double)-0.5) # torch.zeros(B//2,B//2) # 
            A_off = (torch.rand(B//2,B//2,dtype=torch.double)-0.5) # torch.zeros(B//2,B//2) # 
            self.A_on = nn.Parameter(0.5*(A_on + A_on.transpose(0,1)))
            self.A_off = nn.Parameter(0.5*(A_off + A_off.transpose(0,1)))
        else:
            self.a = 1 # nn.Parameter(torch.ones(1))
            self.b = F_dissoc
            self.mask = mask
            self.A = nn.Parameter(torch.rand(B,B,dtype=torch.double)-0.5)
            

    def forward(self,input):
        positions = input[0]
        h_new = input[1]
        batch_size = h_new.shape[0]
        
        r_ij = torch.stack([positions[n,self.idx_i]-positions[n,self.idx_j] for n in range(batch_size)])
        d_ij = torch.norm(r_ij, dim=2)
        R = d_ij[:,0]

        if self.formula=='N2':
            A = torch.zeros(self.B,self.B)
            A[:self.B//2,:self.B//2] = A[self.B//2:,self.B//2:] = self.A_on
            A[:self.B//2,self.B//2:] = A[self.B//2:,:self.B//2] = self.A_off
            F = torch.einsum('pq,n->npq',A,torch.exp(-self.a*R))
        else:
            F = torch.einsum('pq,n->npq',self.A,torch.exp(-self.a*R))
        
        zeros = torch.zeros(h_new.shape,dtype=torch.double)
        zeros[torch.abs(h_new)>1e-7] = 1 # Warning: can create discontinuities, if a non-zero value of h_new drops below the threshold
        F_new = zeros * self.mask * F + self.b
        if F_new.requires_grad:
            F_new.retain_grad()
        
        # solve Fock equation
        e, U = torch.linalg.eigh(F_new)
        if e.requires_grad:
            e.retain_grad()
        
        # compute electronic energy
        E_e = torch.sum(e[:,:self.N//2],dim=1) + torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:self.N//2],torch.einsum('npq,nqj->npj', h_new, U[:,:,:self.N//2])))
        nuc = torch.sum(torch.stack([self.numbers[k]*self.numbers[l]/torch.norm(positions[:,k]-positions[:,l],dim=1)*a0 
                                     for k in range(self.n_atoms) for l in range(k)],dim=-1),dim=-1)
        return F_new, e, E_e + nuc


'''
Functions
'''

def rho(x,y,z,D,numbers,positions,orbitals,basis_func,B,V=None):
    if type(V)==None:
        V=np.identity(B) # rotating coordinate system
    x = V @ np.array([x,y,z])

    def j(k,i,l,m):
        n_k = int(np.sum([np.sum(2*np.array(orbitals[Z])+1) for Z in numbers[:k]])) # number of basis functions
        n_i = np.sum([(2*np.array(orbitals[numbers[k]])+1)[:i]])
        return n_k + n_i + m+l
    indices = [(k,i,l,m) for k in range(len(numbers)) for i,l in enumerate(orbitals[numbers[k]]) for m in range(-l,l+1)]
    
    on_diagonal = np.sum([D[j(k,i,l,m),j(k,i,l,m)] * (basis_func[numbers[k]][i][1]@Gaussian(np.linalg.norm(x - positions[k])/a0,basis_func[numbers[k]][i][0],l) *\
                    sph_harm_real(x - positions[k],l,m))**2 for k,i,l,m in indices])
    off_diagonal = 2*np.sum([D[i1,i2] *\
                            basis_func[numbers[indices[i1][0]]][indices[i1][1]][1]@Gaussian(np.linalg.norm(x - positions[indices[i1][0]])/a0,basis_func[numbers[indices[i1][0]]][indices[i1][1]][0],indices[i1][2]) * sph_harm_real(x - positions[indices[i1][0]],indices[i1][2],indices[i1][3]) *\
                            basis_func[numbers[indices[i2][0]]][indices[i2][1]][1]@Gaussian(np.linalg.norm(x - positions[indices[i2][0]])/a0,basis_func[numbers[indices[i2][0]]][indices[i2][1]][0],indices[i2][2]) * sph_harm_real(x - positions[indices[i2][0]],indices[i2][2],indices[i2][3])
                            for i1 in range(len(indices)) for i2 in range(i1)])
    return on_diagonal + off_diagonal
    # return np.sum([D[j(k1,i1,l1,m1),j(k2,i2,l2,m2)] *\
    #                basis_func[numbers[k1]][i1][1]@Gaussian(np.linalg.norm(x - positions[k1])/a0,basis_func[numbers[k1]][i1][0],l1) * sph_harm_real(x - positions[k1],l1,m1) *\
    #                basis_func[numbers[k2]][i2][1]@Gaussian(np.linalg.norm(x - positions[k2])/a0,basis_func[numbers[k2]][i2][0],l2) * sph_harm_real(x - positions[k2],l2,m2)
    #                for k1 in range(len(numbers)) for i1,l1 in enumerate(orbitals) for m1 in range(-l1,l1+1)
    #                for k2 in range(len(numbers)) for i2,l2 in enumerate(orbitals) for m2 in range(-l2,l2+1)])

def half_factorial(n):
    p=1
    for i in range(0,n+1,2):
        p *= max(n-i,1)
    return p

def Gaussian(r,a,l):
    return 2*(2*a)**0.75/np.pi**0.25*np.sqrt(2**l/half_factorial(2*l+1))*(np.sqrt(2*a)*r)**l*np.exp(-a*r**2)

def phi(x,y,z,c,numbers,positions,orbitals,basis_func,B):
    x = np.array([x,y,z])
    
    def j(k,i,l,m,Z):
        n_k = int(np.sum([np.sum(2*np.array(orbitals[Z])+1) for Z in numbers[:k]])) # number of basis functions
        n_i = np.sum([(2*np.array(orbitals[Z])+1)[:i]])
        return n_k + n_i + m+l
    
    result = np.sum([ c[j(k,i,l,m,Z)] * basis_func[Z][i][1]@Gaussian(np.linalg.norm(x - positions[k])/a0,basis_func[Z][i][0],l) * sph_harm_real(x - positions[k],l,m) for k,Z in enumerate(numbers) for i,l in enumerate(orbitals[Z]) for m in range(-l,l+1) ])
    # j = 0
    # for k, Z_k in enumerate(numbers):
    #     x_k = x - positions[k]
    #     r = np.linalg.norm(x_k)
    #     for i, l in enumerate(orbitals[Z_k]):
    #         a, d = basis_func[Z_k][i]
    #         for m in range(-l,l+1):
    #             result += c[j] * d @ Gaussian(r/a0,a,l) * sph_harm_real(x_k,l,m) # / np.sqrt(quad(lambda r: (d @ Gaussian(r,a,l))**2**r**2,0,np.inf)[0])
    #             j += 1

    return result

def sph_harm_real(x,l,m):
    r = np.linalg.norm(x)
    rho = np.linalg.norm(x[:-1])
    if rho==0:
        phi = 0
    elif x[1]>0:
        phi = np.arccos(x[0]/rho)
    else:
        phi = -np.arccos(x[0]/rho)

    theta = np.arccos(x[2]/r)
    
    if l==1:
        if m == -1:
            m = 1 # px
        elif m == 0:
            m = -1 # py
        else:
            m = 0 # pz

    if m < 0:
        return np.real((sph_harm(m,l,phi,theta) - (-1)**m * sph_harm(-m,l,phi,theta))*1j/np.sqrt(2))
    elif m == 0:
        return np.real(sph_harm(0,l,phi,theta))
    else:
        return np.real((sph_harm(-m,l,phi,theta) + (-1)**m * sph_harm(m,l,phi,theta))/np.sqrt(2))

def Lennard_Jones(r):
    eps = 1.1273427
    sigma = 1.09/2**(1/6)
    return 4*eps*(sigma/r)**6*((sigma/r)**6-1) + eps

def Mie_potential(r,n,c_n,c_m):
    m=6
    return c_n/r**n - c_m/r**m


'''
importing data
'''


def import_raw_data():
     qm9tut = './code/qm9tut'
     qm9data = QM9(
          './code/qm9.db',
          batch_size=100,
          num_train=1000,
          num_val=1000,
          transforms=[
               trn.ASENeighborList(cutoff=5.),
               trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
               trn.CastTo32()
          ],
          property_units={QM9.U0: 'eV'},
          num_workers=1,
          split_file=os.path.join(qm9tut, "split.npz"),
          pin_memory=False, # set to false, when not using a GPU
          load_properties=[QM9.U0], #only load U0 property
     )
     qm9data.prepare_data()
     qm9data.setup()
     return qm9data


def import_data(dataset_size):
     data = connect('qm9.db')
     molecules = []
     formulas = []

     if dataset_size == 'all':
          selected_data = data.select()
          for row in selected_data:
               molecules.append([row.numbers, row.positions])
               formulas.append(row.formula)
     else:
          for i in list(range(dataset_size)):
               for row in data.select(i+1):
                    molecules.append([row.numbers, row.positions])
                    formulas.append(row.formula)
     return molecules, formulas


def import_best_model():
     qm9tut = './qm9tut'
     best_model = torch.load(os.path.join(qm9tut, 'qm9_new_schnet_5_ssp/best_model'), map_location="cpu")
     return best_model


def build_atomwise_model(best_model):
    # unwrap model
    input_modules = best_model.input_modules
    schnet = best_model.representation
    old_output_module = best_model.output_modules[0]

    new_output_module = Atomwise(activation=shifted_softplus,
                                 n_in=128,
                                 output_key=QM9.U0)
    for i in range(2):
        new_output_module.outnet[i].weight = old_output_module.outnet[i].weight
        new_output_module.outnet[i].bias = old_output_module.outnet[i].bias

    #rebuild model
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=input_modules,
        output_modules=[new_output_module]
    )

    return nnpot


def build_model_up_to_layer_t(best_model,n_interactions):
    n_atom_basis = 128

    #get weights
    input_modules = best_model.input_modules
    radial_basis = best_model.representation.radial_basis
    cutoff_fn = best_model.representation.cutoff_fn
    embedding_weights = best_model.representation.embedding.weight
    
    #build new interaction layers
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn
    )
    output_module = extract_output_keys()

    #build new model
    new_model = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=input_modules,
        output_modules=[output_module]
    )
    
    #copy the weights of the embedding and interaction layers from best model to new model
    new_model.representation.embedding.weight = embedding_weights
    for i in range(n_interactions):
        new_model.representation.interactions[i].in2f.weight = best_model.representation.interactions[i].in2f.weight
        new_model.representation.interactions[i].in2f.bias = best_model.representation.interactions[i].in2f.bias
        new_model.representation.interactions[i].f2out[0].weight = best_model.representation.interactions[i].f2out[0].weight
        new_model.representation.interactions[i].f2out[0].bias = best_model.representation.interactions[i].f2out[0].bias
        new_model.representation.interactions[i].f2out[1].weight = best_model.representation.interactions[i].f2out[1].weight
        new_model.representation.interactions[i].f2out[1].bias = best_model.representation.interactions[i].f2out[1].bias
        new_model.representation.interactions[i].filter_network[0].weight = best_model.representation.interactions[i].filter_network[0].weight
        new_model.representation.interactions[i].filter_network[0].bias = best_model.representation.interactions[i].filter_network[0].bias
        new_model.representation.interactions[i].filter_network[1].weight = best_model.representation.interactions[i].filter_network[1].weight
        new_model.representation.interactions[i].filter_network[1].bias = best_model.representation.interactions[i].filter_network[1].bias

    return new_model


def build_molecule_and_make_reference_calculations(molecule_number,translation,basis):
    qm9tut = 'qm9tut'

    #import best model
    best_model = torch.load(os.path.join(qm9tut, 'qm9_new_schnet_5_ssp/best_model'), map_location="cpu")
    atomwise_model = build_atomwise_model(best_model)
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)

    #import molecule data
    qm9data = QM9(
        'qm9.db',
        batch_size=100,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=False, # set to false, when not using a GPU
        load_properties=[QM9.U0], #only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
    
    atomrefs = qm9data.train_dataset.atomrefs[QM9.U0]
    mean = -4.2427 # eV

    data = connect('qm9.db')
    if molecule_number==-2:
        #H2
        numbers, positions = [1,1], np.array([[0,0,0],[0,0,1.4*a0]]) + translation
        A = len(numbers)
        atoms = Atoms(numbers=numbers, positions=positions)
        inputs = converter(atoms)
        predictions = atomwise_model(inputs)
        predictions = [float(predictions[QM9.U0][i,0]) for i in range(A)]
        reference_energy = -31.9306 # eV
    elif molecule_number==-1:
        #O2
        numbers, positions = [8,8], np.array([[0,0,0],[0,0,1.21]]) + translation
        A = len(numbers)
        atoms = Atoms(numbers=numbers, positions=positions)
        inputs = converter(atoms)
        predictions = atomwise_model(inputs)
        predictions = [float(predictions[QM9.U0][i,0]) for i in range(A)]
        reference_energy = -4090.38 # eV
    elif molecule_number==0:
        #N2
        numbers, positions = [7,7], np.array([[0,0,0],[0,0,1.1]]) + translation
        A = len(numbers)
        atoms = Atoms(numbers=numbers, positions=positions)
        inputs = converter(atoms)
        predictions = atomwise_model(inputs)
        predictions = [float(predictions[QM9.U0][i,0]) for i in range(A)]
        reference_energy = -3007.61 # eV
    else:
        for i in [molecule_number]:
            for row in data.select(i):
                print(row.formula)
                numbers, positions = row.numbers, row.positions + translation
                A = len(numbers)
                atoms = Atoms(numbers=numbers, positions=positions)
                inputs = converter(atoms)
                predictions = atomwise_model(inputs)
                predictions = [float(predictions[QM9.U0][j,0]) for j in range(A)]
                reference_energy = row.data["energy_U0"][0]*27.2114
                print(reference_energy)
    
    # build molecule and get parameters
    mol = gto.M(
        atom = [[atom_symbols[numbers[i]-1],positions[i]] for i in range(A)],
        basis = basis
    )
    M = mol.nao_nr()
    N = mol.nelectron

    atomic_predictions = np.zeros(A)
    for k in range(A):
        atomic_predictions[k] = (atomrefs[numbers[k]] + mean + predictions[k])/27.2114 # in atomic units
    print('Prediction best model in a.u.:', atomic_predictions)
    print(sum(atomic_predictions)*27.2114)
    
    # make reference calculation
    ovlp = mol.intor('int1e_ovlp')
    kin = mol.intor('int1e_kin')
    nuc = mol.intor('int1e_nuc')
    v = mol.intor('int2e')
    rhf = scf.RHF(mol)
    rhf.kernel()
    c0 = np.transpose(rhf.mo_coeff)[:(N//2),:] # better: get initial guess
    #print(c0)
    c0 = c0.reshape((N//2)*M)
    #print(energy_RHF(np.transpose(c_HF)[:(N//2),:])) # reproduces HF energy
    return mol,A,N,M,numbers,positions,ovlp,kin,nuc,v,atomic_predictions,reference_energy,c0


def compute_nuclear_repulsion(k,numbers,positions):
    #distances = squareform(pdist(positions))
    distances = np.array([np.linalg.norm(positions[k]-positions[l]) for l in range(len(positions))])
    neighbors = numbers[(0 < distances) & (distances < 5)]
    neighbor_distances = distances[(0 < distances) & (distances < 5)]
    return 0.5*numbers[k]*np.sum(np.divide(neighbors,neighbor_distances))*a0


'''
overlap integrals
'''

def integral(n):
    if n%2 == 1:
        return 0
    else:
        return factorial(n)/4**(n//2)/factorial(n//2)*np.sqrt(np.pi) # \Gamma(n/2+1/2)
    
def Bi(n,k):
    if k<0 or k>n:
        return 0
    else:
        return binom(n,k)

def x_cartesian_integral(p,q,r,s,a1,a2,a3,a4,Q,PR1,PR2,PR3,PR4,d):
    p,q,r,s = int(p),int(q),int(r),int(s)
    b = a1 + a2 + a3 + a4
    return np.exp(-Q[d]) * np.sum([b**(-(i+j+k+l+1)/2)*Bi(p,i)*Bi(q,j)*Bi(r,k)*Bi(s,l)*\
                                   PR1[d]**(p-i)*PR2[d]**(q-j)*PR3[d]**(r-k)*PR4[d]**(s-l)*integral(i+j+k+l) 
                                   for i in range(p+1) for j in range(q+1) for k in range(r+1) for l in range(s+1)])

def C(l,m,t,u,w):
    return (-1)**(t+w-v(m))*0.25**t * Bi(l,t) * Bi(l-t,abs(m)+t) * Bi(t,u) * Bi(abs(m),2*w)

def N(l,m):
    if m==0:
        return np.sqrt(factorial(l+abs(m))*factorial(l-abs(m)))/2**abs(m)/factorial(l)
    else:
        return np.sqrt(2*factorial(l+abs(m))*factorial(l-abs(m)))/2**abs(m)/factorial(l)

def v(m):
    if m<0:
        return 0.5
    else:
        return 0

def half_factorial(n):
    p=1
    for i in range(0,n+1,2):
        p *= max(n-i,1)
    return p

def GTO_norm(a,l):
    return 2*(2*a)**0.75/np.pi**0.25*np.sqrt(2**l/half_factorial(2*l+1))*np.sqrt(2*a)**l * np.sqrt((2*l+1)/4/np.pi)

indices_c_int = [[(t,u,w) for t in range((l-abs(m))//2+1) for u in range(t+1) for w in np.arange( v(m),np.floor(abs(m)/2-v(m))+v(m)+1,step=1 )]
           for l in range(2) for m in range(-l,l+1)]

def spherical_integral(l1,m1,l2,m2,l3,m3,l4,m4,a1,a2,a3,a4,R1,R2,R3,R4):
    Q = np.array([(a1*a2*(R1[d]-R2[d])**2 + a1*a3*(R1[d]-R3[d])**2 + a1*a4*(R1[d]-R4[d])**2 +\
                    a2*a3*(R2[d]-R3[d])**2 + a2*a4*(R2[d]-R4[d])**2 + a3*a4*(R3[d]-R4[d])**2 )/(a1+a2+a3+a4) for d in range(3)])
    P = (a1*R1 + a2*R2 + a3*R3 + a4*R4)/(a1+a2+a3+a4)
    PR1 = P - R1
    PR2 = P - R2
    PR3 = P - R3
    PR4 = P - R4
    result = 0
    for t1 in range((l1-abs(m1))//2+1):
        for u1 in range(t1+1):
            for v1 in np.arange( v(m1),np.floor(abs(m1)/2-v(m1))+v(m1)+1,step=1 ):
                for t2 in range((l2-abs(m2))//2+1):
                    for u2 in range(t2+1):
                        for v2 in np.arange( v(m2),np.floor(abs(m2)/2-v(m2))+v(m2)+1,step=1 ):
                            for t3 in range((l3-abs(m3))//2+1):
                                for u3 in range(t3+1):
                                    for v3 in np.arange( v(m3),np.floor(abs(m3)/2-v(m3))+v(m3)+1,step=1 ):
                                        for t4 in range((l4-abs(m4))//2+1):
                                            for u4 in range(t4+1):
                                                for v4 in np.arange( v(m4),np.floor(abs(m4)/2-v(m4))+v(m4)+1,step=1 ):
                                                    result += C(l1,m1,t1,u1,v1)*C(l2,m2,t2,u2,v2)*C(l3,m3,t3,u3,v3)*C(l4,m4,t4,u4,v4) *\
                                                            x_cartesian_integral(2*t1+abs(m1)-2*(u1+v1),2*t2+abs(m2)-2*(u2+v2),2*t3+abs(m3)-2*(u3+v3),2*t4+abs(m4)-2*(u4+v4),a1,a2,a3,a4,Q,PR1,PR2,PR3,PR4,0) *\
                                                            x_cartesian_integral(2*(u1+v1),2*(u2+v2),2*(u3+v3),2*(u4+v4),a1,a2,a3,a4,Q,PR1,PR2,PR3,PR4,1) *\
                                                            x_cartesian_integral(l1-2*t1-abs(m1),l2-2*t2-abs(m2),l3-2*t3-abs(m3),l4-2*t4-abs(m4),a1,a2,a3,a4,Q,PR1,PR2,PR3,PR4,2)
    
    return N(l1,m1)*N(l2,m2)*N(l3,m3)*N(l4,m4)*result

def spherical_integral_1D(a1,a2,l):
    m = 0
    Null = np.array([0,0,0])
    result = 0
    for t1 in range((l-abs(m))//2+1):
        for u1 in range(t1+1):
            for v1 in np.arange( v(m),np.floor(abs(m)/2-v(m))+v(m)+1,step=1 ):
                for t2 in range((l-abs(m))//2+1):
                    for u2 in range(t2+1):
                        for v2 in np.arange( v(m),np.floor(abs(m)/2-v(m))+v(m)+1,step=1 ):
                            for t3 in range((l-abs(m))//2+1):
                                for u3 in range(t3+1):
                                    for v3 in np.arange( v(m),np.floor(abs(m)/2-v(m))+v(m)+1,step=1 ):
                                        for t4 in range((l-abs(m))//2+1):
                                            for u4 in range(t4+1):
                                                for v4 in np.arange( v(m),np.floor(abs(m)/2-v(m))+v(m)+1,step=1 ):
                                                    result += C(l,m,t1,u1,v1)*C(l,m,t2,u2,v2)*C(l,m,t3,u3,v3)*C(l,m,t4,u4,v4) *\
                                                            x_cartesian_integral(2*t1+abs(m)-2*(u1+v1),2*t2+abs(m)-2*(u2+v2),2*t3+abs(m)-2*(u3+v3),2*t4+abs(m)-2*(u4+v4),a1,a2,0,0,Null,Null,Null,Null,Null,0) *\
                                                            x_cartesian_integral(2*(u1+v1),2*(u2+v2),2*(u3+v3),2*(u4+v4),a1,a2,0,0,Null,Null,Null,Null,Null,1) *\
                                                            x_cartesian_integral(l-2*t1-abs(m),l-2*t2-abs(m),l-2*t3-abs(m),l-2*t4-abs(m),a1,a2,0,0,Null,Null,Null,Null,Null,2)
    
    return result * GTO_norm(a1,l)*GTO_norm(a2,l)

def contracted_integral(l1,m1,l2,m2,l3,m3,l4,m4,a1,a2,a3,a4,d1,d2,d3,d4,R1,R2,R3,R4):
    # formula validated for two centers
    return np.sum([d1[i] * d2[j] * d3[k] * d4[l] *\
                   spherical_integral(l1,m1,l2,m2,l3,m3,l4,m4,a1[i],a2[j],a3[k],a4[l],R1,R2,R3,R4)*GTO_norm(a1[i],l1)*GTO_norm(a2[j],l2)*GTO_norm(a3[k],l3)*GTO_norm(a4[l],l4)
                   for i in range(len(a1)) for j in range(len(a2)) for k in range(len(a3)) for l in range(len(a4))])

def contracted_integral_norm(l1,m1,l2,m2,l3,m3,l4,m4,a1,a2,a3,a4,d1,d2,d3,d4,R1,R2,R3,R4):
    # formula validated for two centers
    return np.sum([d1[i] * d2[j] * d3[k] * d4[l] *\
                   spherical_integral(l1,m1,l2,m2,l3,m3,l4,m4,a1[i],a2[j],a3[k],a4[l],R1,R2,R3,R4)*GTO_norm(a1[i],l1)*GTO_norm(a2[j],l2) #*GTO_norm(a3[k],l3)*GTO_norm(a4[l],l4)
                   for i in range(len(a1)) for j in range(len(a2)) for k in range(len(a3)) for l in range(len(a4))])

def compute_four_center_overlap_tensor(formula,basis,numbers,lo,hi,increments):
    basis_func, orbitals = get_GTO_coeff(basis,numbers)
    orbitals = orbitals[numbers[0]]
    # get number of atoms and basis functions
    mol = gto.M(atom = [[numbers[0],[0,0,0]],[numbers[1],[1,0,0]]],
                basis = basis,
                unit='Angstrom')
    B = mol.nao_nr()
    M = mol.natm

    # compute tensor
    indices = [(k,i,l,m) for k in range(M) for i,l in enumerate(orbitals) for m in range(l,-l-1,-1)]
    for R in np.linspace(lo,hi,increments):
        positions = [np.array([0,0,0])/a0, np.array([R,0,0])/a0] # in atomic units

        S4 = np.zeros((B,B,B,B))
        start = time()
        for j1 in range(len(indices)):
            for j2 in range(j1+1):
                for j3 in range(j2+1):
                    for j4 in range(j3+1):
                        k1,i1,l1,m1 = indices[j1]
                        k2,i2,l2,m2 = indices[j2]
                        k3,i3,l3,m3 = indices[j3]
                        k4,i4,l4,m4 = indices[j4]
                        
                        value = contracted_integral(l1,m1,l2,m2,l3,m3,l4,m4,
                                                    basis_func[numbers[k1]][i1][0],basis_func[numbers[k2]][i2][0],basis_func[numbers[k3]][i3][0],basis_func[numbers[k4]][i4][0],
                                                    basis_func[numbers[k1]][i1][1],basis_func[numbers[k2]][i2][1],basis_func[numbers[k3]][i3][1],basis_func[numbers[k4]][i4][1],
                                                    positions[k1],positions[k2],positions[k3],positions[k4])
                        for J in list(set(permutations([j1,j2,j3,j4]))):
                            S4[J] = value
                        
        stop = time()
        print(np.round(stop-start,2),"s")
        torch.save(torch.tensor(S4,dtype=torch.float),"temp/{}_four_center_ovlp_{}_R_{}".format(formula,basis,round(R*1000)))

'''
Creating datasets
'''

def create_x_and_predictions(molecules,best_model,cut_open_model,incorporate_nuc_in_V: bool):
     # rebuild model atom-wise
     atomwise_model = build_atomwise_model(best_model)
     
     N = sum([len(molecule[0]) for molecule in molecules]) # sample size
     x = torch.zeros((N,n_features))
     nuc = torch.zeros((N,1))
     numbers = torch.zeros((N,1))
     E = torch.zeros((N,1))
     converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=5.)) # dtype=torch.float32
     counter = 0
     with torch.no_grad():
          for molecule in molecules:
               A = len(molecule[0])
               numbers_i = molecule[0]
               positions = molecule[1]
               atomic_energies = torch.zeros((A,1))
               nuclear_repulsion = torch.zeros(A)
               for k in range(A):
                    atomic_energies[k,0] = atomrefs[numbers_i[k]] # in eV
                    if not incorporate_nuc_in_V:
                         nuclear_repulsion[k] = compute_nuclear_repulsion(k,numbers_i,positions) # in a.u.
               atoms = Atoms(numbers=numbers_i, positions=positions)
               inputs = converter(atoms)    
               x_k = cut_open_model(inputs)["scalar_representation"] # shape (A x n_features)
               E_pred_k = atomwise_model(inputs)["energy_U0"] # in eV
               
               x[counter:counter+A,:] = x_k
               nuc[counter:counter+A,0] = nuclear_repulsion
               numbers[counter:counter+A,0] = torch.tensor(numbers_i)
               E[counter:counter+A,:] = (E_pred_k + atomic_energies + mean)/Hartree_to_eV # in a.u.
               counter += A

     return SchNet_XAIDataSet(E,numbers,nuc,x)

'''
Dataset classes
'''

# class SchNet_XAIDataSet(torch.utils.data.Dataset):
#     def __init__(self, energies, numbers, nuc, x):
#         # convert atomic numbers from float to int
#         self.numbers = torch.tensor(numbers.reshape(numbers.shape[0]),dtype=torch.int)

#         # create input and target data
#         self.target_energies = energies # shape (sample_size, 1)
#         self.input_data = torch.hstack((numbers,nuc,x)) # (sample_size, n_features+2))

#     def __len__(self):
#         return len(self.target_energies)

#     def __getitem__(self, k):
#         inputs = self.input_data[k,:]
#         target = self.target_energies[k,:]
#         return inputs, target

class SchNet_XAIDataSet_Fock_matrices(torch.utils.data.Dataset):
    def __init__(self, energies, Fock_matrices, numbers, nuc, x):
        self.numbers = torch.tensor(numbers.reshape(numbers.shape[0]),dtype=torch.int)
        self.SchNet_energies = energies # shape (sample_size, 1)
        self.reference_matrices = Fock_matrices
        self.input_data = torch.hstack((numbers,nuc,x)) # (sample_size, n_features+2))

    def __len__(self):
        return len(self.SchNet_energies)

    def __getitem__(self, k):
        inputs = self.input_data[k,:]
        E = self.SchNet_energies[k,:]
        F_a, F_b = self.reference_matrices[self.numbers[k]-1]
        target = (E,F_a,F_b,self.numbers)
        return inputs, target


'''
Export cube files
'''

from pyscf import lib, __config__
from pyscf.pbc.gto import Cell
from pyscf.tools.cubegen import Cube
from pyscf.dft import numint

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)

def _get_coords(self,V) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        frac_coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
        real_coords = frac_coords @ self.box + self.boxorig # Convert fractional coordinates to real-space coordinates
        rot_coords = real_coords @ V
        return rot_coords

def density(mol, outfile, dm, V, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
    """Calculates electron density and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    
    Cube.get_coords = _get_coords # override method to allow for coordinate rotation
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    extent = np.array([14,11,4]) # np.max(coord, axis=0) - np.min(coord, axis=0) + 2*margin
    cc.box = np.diag(extent)

    # if origin is not supplied, set it as the minimum coordinate minus a margin.
    #origin = np.min(coord, axis=0) - margin
    cc.boxorig = np.array([-7,-5,-2]) # np.asarray(origin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords(V)
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho


def orbital(mol, outfile, coeff, V, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
    """Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    
    Cube.get_coords = _get_coords # override method to allow for coordinate rotation
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    extent = np.array([14,11,4]) # np.max(coord, axis=0) - np.min(coord, axis=0) + 2*margin
    cc.box = np.diag(extent)

    # if origin is not supplied, set it as the minimum coordinate minus a margin.
    cc.boxorig = np.array([-7,-5,-2])

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords(V)
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = np.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = np.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return orb_on_grid


def export_cube_file(model,formula,basis,numbers,label,database):
    '''
    Exports a cube file for the reference density, the xPaiNN density and the difference xPaiNN density minus reference density
    '''
    positions = database.get_properties(label)[1]["_positions"]
    F_ref = database.get_properties(label)[1]["hamiltonian"]
    S_ref = database.get_properties(label)[1]["overlap"]
    E_ref = database.get_properties(label)[1]["energy"][0]

    # center of mass as origin
    center_of_mass = np.sum(positions*numbers[:,None],axis=0)/sum(numbers)
    positions -= center_of_mass

    mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
    N = mol.nelectron
    n_atoms = mol.natm
    
    # get rotation matrix into principle axes
    m = numbers
    R = positions
    theta_n = [m[n]*np.array([[R[n,1]**2+R[n,2]**2, -R[n,0]*R[n,1], -R[n,0]*R[n,2]],
                              [-R[n,0]*R[n,1], R[n,0]**2+R[n,2]**2, -R[n,1]*R[n,2]],
                              [-R[n,0]*R[n,2], -R[n,1]*R[n,2], R[n,0]**2+R[n,1]**2]]) for n in range(n_atoms)]
    inertia_tensor = np.sum(np.stack(theta_n),axis=0)
    I, V = np.linalg.eigh(inertia_tensor)
    if (positions@V)[-3,1] < 0:
        V *= -1
    
    # calculate reference orbitals
    _, P_ref = scipy.linalg.eigh(F_ref,S_ref)
    
    # re-order orbitals in PySCF format
    basisdef = torch.tensor(database.get_metadata('basisdef'))
    orb_indices = []
    for Z in numbers:
        for i in range(14):
            if basisdef[Z,i,2]>0:
                orb_indices.append((Z,i))
    tmp = np.zeros(P_ref.shape)
    for i, ind in enumerate(orb_indices):
        if basisdef[ind+(3,)]==1:
            if basisdef[ind+(4,)]==-1: # pz
                tmp[i+2] = P_ref[i,:]
            elif basisdef[ind+(4,)]==0: # px
                tmp[i-1] = P_ref[i,:]
            elif basisdef[ind+(4,)]==1: # py
                tmp[i-1] = P_ref[i,:]
        elif basisdef[ind+(3,)]==2:
            if basisdef[ind+(4,)]==-2:
                tmp[i+2] = P_ref[i,:]
            elif basisdef[ind+(4,)]==-1:
                tmp[i+2] = P_ref[i,:]
            elif basisdef[ind+(4,)]==0:
                tmp[i-1] = P_ref[i,:]
            elif basisdef[ind+(4,)]==1:
                tmp[i+1] = P_ref[i,:]
            elif basisdef[ind+(4,)]==2:
                tmp[i-4] = P_ref[i,:]
        else:
            tmp[i] = P_ref[i,:]
    P_ref = tmp
    D_ref = P_ref[:,:N//2] @ P_ref[:,:N//2].transpose()
    print(E_ref)

    # make orbital predictions
    F_new, e, E = model((torch.tensor(positions).reshape((1,n_atoms,3)),None))
    e, U = torch.linalg.eigh(F_new[0])
    S = get_h_and_S(basis,numbers,positions)[1].detach().numpy()
    S_inv_root = scipy.linalg.sqrtm(np.linalg.inv(S)).real
    P = S_inv_root @ U.detach().numpy()
    D = P[:,:N//2] @ P[:,:N//2].transpose()
    print(float(E))
    
    # create and save cube files
    density(mol, '{}_{}_{}_true_density.cub'.format(formula,basis,label), D_ref, V.transpose(),nx=150,ny=150,nz=150)
    density(mol, '{}_{}_{}_xPaiNN_v4_density.cub'.format(formula,basis,label), D, V.transpose(),nx=150,ny=150,nz=150)
    density(mol, '{}_{}_{}_xPaiNN_v4_density_diff.cub'.format(formula,basis,label), D-D_ref, V.transpose(),nx=150,ny=150,nz=150)
    #for j in range(N//2):
        #cubegen.orbital(mol, '{}_{}_{}_true_orbital_{}.cub'.format(formula,basis,label,j), P_ref[:,j])
        #cubegen.orbital(mol, '{}_{}_{}_xPaiNN_v4_orbital_{}.cub'.format(formula,basis,label,j), P[:,j])
    return

def plot_proton_transfer(model,formula,basis,numbers,cut,increments=150,loc=False):
    from ase.io import read
    from pyscf import gto
    from schnetpack.interfaces.ase_interface import AtomsConverter

    # import reference orbitals and positions
    F_ref = np.load('reference_data_proton_transfer/F.npy')
    S_ref = np.load('reference_data_proton_transfer/S.npy')
    ats = read('reference_data_proton_transfer/mda_extracted_88300_88800.xyz', index=":")
    converter = AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=cut),
            device="cpu",
            dtype=torch.float32
        )

    # export cube file for densities, difference density and orbitals for each sample point at a time
    for i, at in enumerate(ats):
        atom = converter(at)
        positions = atom["_positions"].detach().numpy()
        Fi_ref = F_ref[i]
        Si_ref = S_ref[i]

        # center of mass as origin
        center_of_mass = np.sum(positions*numbers[:,None],axis=0)/sum(numbers)
        positions -= center_of_mass

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        N = mol.nelectron
        n_atoms = mol.natm

        # get rotation matrix into principle axes
        m = numbers
        R = positions
        theta_n = [m[n]*np.array([[R[n,1]**2+R[n,2]**2, -R[n,0]*R[n,1], -R[n,0]*R[n,2]],
                                    [-R[n,0]*R[n,1], R[n,0]**2+R[n,2]**2, -R[n,1]*R[n,2]],
                                    [-R[n,0]*R[n,2], -R[n,1]*R[n,2], R[n,0]**2+R[n,1]**2]]) for n in range(n_atoms)]
        inertia_tensor = np.sum(np.stack(theta_n),axis=0)
        I, V = np.linalg.eigh(inertia_tensor)
        if (positions@V)[-3,1] < 0:
            V *= -1

        # calculate reference orbitals
        _, P_ref = scipy.linalg.eigh(Fi_ref,Si_ref)
        D_ref = P_ref[:,:N//2] @ P_ref[:,:N//2].transpose()
        
        # make orbital predictions
        F_new, e, E = model((torch.tensor(positions).reshape((1,n_atoms,3)),None))
        e, U = torch.linalg.eigh(F_new[0])
        S = get_h_and_S(basis,numbers,positions)[1].detach().numpy()
        S_inv_root = scipy.linalg.sqrtm(np.linalg.inv(S)).real
        P = S_inv_root @ U.cpu().detach().numpy()
        D = P[:,:N//2] @ P[:,:N//2].transpose()

        # localize orbitals
        if loc:
            mlo = PM(mol)
            P = mlo.kernel(P[:,:N//2])
            P_ref = mlo.kernel(P_ref[:,:N//2])

        # create and save cube files
        density(mol, 'cube_files/{}_{}_{}_true_density_traj.cub'.format(formula,basis,i), D_ref, V.transpose(),nx=increments,ny=increments,nz=3)
        density(mol, 'cube_files/{}_{}_{}_xPaiNN_v4_density_traj.cub'.format(formula,basis,i), D, V.transpose(),nx=increments,ny=increments,nz=3)
        density(mol, 'cube_files/{}_{}_{}_xPaiNN_v4_density_diff_traj.cub'.format(formula,basis,i), D-D_ref, V.transpose(),nx=increments,ny=increments,nz=3)
        for j in range(N//2):
            orbital(mol, 'cube_files/{}_{}_{}_true_orbital_{}_traj.cub'.format(formula,basis,i,j), P_ref[:,j], V.transpose(),nx=increments,ny=increments,nz=3)
            orbital(mol, 'cube_files/{}_{}_{}_xPaiNN_v4_orbital_{}_traj.cub'.format(formula,basis,i,j), P[:,j], V.transpose(),nx=increments,ny=increments,nz=3)
    
    plane_index=1
    for n in range(501):
        cub_file = 'cube_files/{}_{}_{}_true_density_traj.cub'.format(formula,basis,n)
        save_name = 'images/{}_{}_{}_true_density_traj.png'.format(formula,basis,n,j)
        plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type='density')
        cub_file = 'cube_files/{}_{}_{}_xPaiNN_v4_density_traj.cub'.format(formula,basis,n)
        save_name = 'images/{}_{}_{}_xPaiNN_v4_density_traj.png'.format(formula,basis,n,j)
        plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type='density')
        cub_file = 'cube_files/{}_{}_{}_xPaiNN_v4_density_diff_traj.cub'.format(formula,basis,n)
        save_name = 'images/{}_{}_{}_xPaiNN_v4_density_diff_traj.png'.format(formula,basis,n,j)
        plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type='density difference')
        for j in range(N//2):
            cub_file = 'cube_files/{}_{}_{}_true_orbital_{}_traj.cub'.format(formula,basis,n,j)
            save_name = 'images/{}_{}_{}_true_orbital_{}_traj.png'.format(formula,basis,n,j)
            plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type='orbital')
            cub_file = 'cube_files/{}_{}_{}_xPaiNN_v4_orbital_{}_traj.cub'.format(formula,basis,n,j)
            save_name = 'images/{}_{}_{}_xPaiNN_v4_orbital_{}_traj.png'.format(formula,basis,n,j)
            plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type='orbital')
    return

def plot_cube_file(cub_file,save_name,numbers,plane_index,plotting_type,plane='xy'):
    '''
    not fully finished for xz and yz planes
    '''
    n_atoms = len(numbers)
    skip = 5 + n_atoms # lines containing meta-information

    # import values of the chosen plane
    f = open(cub_file)
    contents = csv.reader(f, delimiter=' ')
    j=0
    row = []
    positions = np.zeros((n_atoms,3))
    if plane=='yz':
        for i,entries in enumerate(contents):
            if i<2:
                continue
            if i==2:
                row = [float(entry) for entry in entries if entry!='']
                origin = np.array(row[1:])
                continue
            if i==3:
                row = [entry for entry in entries if entry!='']
                x_incr = float(row[1])
                continue
            if i==4:
                row = [entry for entry in entries if entry!='']
                ny = int(row[0])
                y_incr = float(row[2])
                continue
            if i==5:
                row = [entry for entry in entries if entry!='']
                nz = int(row[0])
                z_incr = float(row[3])
                continue
            if i>5 and i<5+n_atoms+1:
                row = [entry for entry in entries if entry!='']
                positions[i-6] = np.array(row[2:])
                density = np.zeros((nx,ny))
                lines = nz//6 + (nz%6>0)
                continue

            if i>skip+lines*ny*plane_index and i<lines*ny*(plane_index+1)+16:
                if j<lines:
                    j+=1
                    row.append(np.array([entry for entry in entries if entry!='']))
                else:
                    row = np.hstack(row)
                    density[(i-skip-lines*ny*plane_index)//lines-1,:] = row
                    j=1
                    row = []
                    row.append(np.array([entry for entry in entries if entry!='']))

    elif plane=='xz':
        k=0
        for i,entries in enumerate(contents):
            if i<2:
                continue
            if i==2:
                row = [float(entry) for entry in entries if entry!='']
                origin = np.array(row[1:])
                continue
            if i==3:
                row = [entry for entry in entries if entry!='']
                nx = int(row[0])
                x_incr = float(row[1])
                continue
            if i==4:
                row = [entry for entry in entries if entry!='']
                ny = int(row[0])
                y_incr = float(row[2])
                continue
            if i==5:
                row = [entry for entry in entries if entry!='']
                nz = int(row[0])
                z_incr = float(row[3])
                continue
            if i>5 and i<5+n_atoms+1:
                row = [entry for entry in entries if entry!='']
                positions[i-6] = np.array(row[2:])
                row=[]
                density = np.zeros((nx,ny))
                lines = nz//6 + (nz%6>0)
                continue

            if (i-skip-1)//lines%ny==plane_index:
                j+=1
                row.append(np.array([entry for entry in entries if entry!='']))
            if j==lines:
                row = np.hstack(row)
                density[k,:] = row
                k+=1
                j=0
                row=[]

    elif plane=='xy':
        k=plane_index//6
        l=plane_index%6
        for i,entries in enumerate(contents):
            if i<2:
                continue
            if i==2:
                row = [float(entry) for entry in entries if entry!='']
                origin = np.array(row[1:])
                continue
            if i==3:
                row = [entry for entry in entries if entry!='']
                nx = int(row[0])
                x_incr = float(row[1])
                continue
            if i==4:
                row = [entry for entry in entries if entry!='']
                ny = int(row[0])
                y_incr = float(row[2])
                continue
            if i==5:
                row = [entry for entry in entries if entry!='']
                nz = int(row[0])
                z_incr = float(row[3])
                continue
            if i>5 and i<5+n_atoms+1:
                row = [entry for entry in entries if entry!='']
                positions[i-6] = np.array(row[2:])
                density = np.zeros((nx,ny))
                lines = nz//6 + (nz%6>0)
                continue
            
            jx = (i-skip-1)//(lines*ny)
            jy = (i-skip-1)//lines%ny
            if (i-skip-1)%lines==k:
                row = [entry for entry in entries if entry!='']
                density[jx,jy] = row[l]
    
    # flip y-axis
    density = np.flip(density,axis=1)
    #print(np.sum(positions*numbers[:,None],axis=0)/sum(numbers))
    # get rotation matrix into principle axes
    m = numbers
    R = positions
    theta_n = [m[n]*np.array([[R[n,1]**2+R[n,2]**2, -R[n,0]*R[n,1], -R[n,0]*R[n,2]],
                              [-R[n,0]*R[n,1], R[n,0]**2+R[n,2]**2, -R[n,1]*R[n,2]],
                              [-R[n,0]*R[n,2], -R[n,1]*R[n,2], R[n,0]**2+R[n,1]**2]]) for n in range(n_atoms)]
    inertia_tensor = np.sum(np.stack(theta_n),axis=0)
    _, V = np.linalg.eigh(inertia_tensor)
    if (positions@V)[-3,1] < 0:
        V *= -1
    positions = positions @ V * a0 # convert to Angstrom
    
    if plane=='yz':
        n1=ny
        n2=nz
        delta1=y_incr
        delta2=z_incr
        index=np.array([1,2])
    elif plane=='xz':
        n1=nx
        n2=nz
        delta1=x_incr
        delta2=z_incr
        index=np.array([0,2])
    elif plane=='xy':
        n1=nx
        n2=ny
        delta1=x_incr
        delta2=y_incr
        index=np.array([0,1])
    
    # plot function
    xmin = origin[index][0]
    xmax = xmin + n1*delta1
    ymin = origin[index][1]
    ymax = ymin + n2*delta2
    vmax = 0.2 # maximal absolute value of the colormap
    if plotting_type=='density difference':
        vmax = 0.025 # maximal absolute value of the colormap
        imshow_kwargs = {
            'vmax': vmax,
            'vmin': -vmax,
            'cmap': 'PRGn',
            'extent': (xmin*a0, xmax*a0, ymin*a0, ymax*a0)
        }
    elif plotting_type=='density':
        vmax = 0.2 # maximal absolute value of the colormap
        imshow_kwargs = {
            #'vmax': vmax,
            #'vmin': 0,
            'cmap': 'Blues',
            'extent': (xmin*a0, xmax*a0, ymin*a0, ymax*a0)
        }
    elif plotting_type=='orbital':
        vmax = 0.2 # maximal absolute value of the colormap
        imshow_kwargs = {
            'vmax': vmax,
            'vmin': -vmax,
            'cmap': 'RdBu',
            'extent': (xmin*a0, xmax*a0, ymin*a0, ymax*a0)
        }
    import matplotlib.colors as clrs
    colors = [None,'grey',None,None,None,None,'k','b','r']
    X = np.linspace(-7*a0,7*a0,150)
    Y = np.linspace(-5*a0,6*a0,150)
    plt.contour(X,Y,np.flip(density,axis=1).transpose(),levels=[0.0001,0.001,0.01,0.1,1],colors="k")
    plt.imshow(density.transpose(),**imshow_kwargs,norm=clrs.LogNorm(vmin=0.01,vmax=0.2))
    #plt.xlabel(r"$x$ position in Angstrom",fontsize=25)
    #plt.ylabel(r"$y$ position in Angstrom",fontsize=25)
    #plt.xticks([-3,-2,-1,0,1,2,3], fontsize=25)
    #plt.yticks([-2,-1,0,1,2,3], fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(bottom=False,left=False)
    cmap = plt.colorbar()
    cmap.ax.tick_params(labelsize=25)
    for k in range(n_atoms):
        plt.scatter([positions[k,index[0]]],[positions[k,index[1]]],s=100,c=colors[numbers[k]]) # plt.scatter([v[0]/delta1],[increments-v[1]/delta2],s=100,c=colors[numbers[k]])
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    plt.show()
    return

def diatomics_orbital_plot(model,model_name,formula,basis,numbers,loc=False):
    """
    Plots a slice of all orbitals and the density of a diatomic molecule through the x-y-plane in its equilibrium 
    configuration/at its equilibrium distance.
    """
    if formula=='N2':
        R_star = 1.0977
    elif formula=='CO':
        R_star = 1.12
    
    # get number of electrons, number of basis functions, number of atoms, GTO parameters
    positions = np.array([[-R_star/2,0,0],[R_star/2,0,0]])
    mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
    N = mol.nelectron
    B = mol.nao_nr()
    M = mol.natm
    basis_func, orbitals = get_GTO_coeff(basis,numbers)
    mf = scf.RHF(mol)
    mf.kernel()

    # get h matrix
    S = mol.intor('int1e_ovlp')
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    S_root = torch.tensor(scipy.linalg.sqrtm(S))
    S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)))
    h_new = (S_inv_root @ h @ S_inv_root).reshape((1,B,B))

    # get predicted orbital coefficients
    positions = torch.tensor(positions).reshape((1,M,3))
    F_new, e, E = model((positions,h_new))
    F_new = F_new.reshape((B,B))
    _, U = torch.linalg.eigh(F_new)
    P = (S_inv_root @ U).detach().numpy()
    
    # localize orbitals
    if loc:
        mlo = PM(mol)
        P = torch.tensor(mlo.kernel(P[:,:N//2]).real)

        # order orbitals according to energy
        e_loc = torch.diag(P.transpose(0,1) @ S_root @ F_new @ S_root @ P)
        _, indices = e_loc.sort()
        P = P[:,indices].detach().numpy()
    print(np.round(P[:,:N//2],3).transpose())

    xmin=-1.5
    xmax=1.5
    ymin=-1
    ymax=1
    xn=200
    yn=100
    values = np.zeros((yn,xn))
    X = np.linspace(xmin,xmax,xn)
    Y = np.linspace(ymin,ymax,yn)

    # plot predicted orbitals
    U0 = P[:,:N//2]
    positions = positions.detach().numpy().reshape((M,3))
    for nu in range(N//2):  
        values_j = np.array([phi(X[i],Y[j],0,U0[:,nu],numbers,positions,orbitals,basis_func,B) for j in range(len(Y)) for i in range(len(X))])
        values_j = values_j.reshape((yn,xn)).real
        values += values_j**2

        vmax = 0.4
        imshow_kwargs = {
            'vmax': vmax,
            'vmin': -vmax,
            'cmap': 'RdBu',
            'extent': (xmin, xmax, ymin, ymax),
        }

        fig, ax = plt.subplots()
        pos = ax.imshow(values_j, **imshow_kwargs)
        # plt.contour(X,Y,values_j)
        for k,Z_k in enumerate(numbers):
            if Z_k == 6:
                color = 'k'
            elif Z_k == 7:
                color = 'b'
            elif Z_k == 8:
                color = 'r'
            plt.scatter([positions[k,0]],[0],s=100,c=color)
        #plt.xlabel(r"$x$ position in Angstrom",fontsize=22)
        #plt.ylabel(r"$y$ position in Angstrom",fontsize=22)
        #plt.xticks([-1.5,-1,-0.5,0,0.5,1,1.5], fontsize=22)
        #plt.yticks([-1,-0.5,0,0.5,1], fontsize=22)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.tick_params(bottom=False,left=False)
        # fig.colorbar(pos)
        if loc:
            plt.savefig("images/{}_{}_{}_orbital_loc_{}_R_{}.png".format(formula,basis,model_name,nu,round(R_star*100)))
        else:
            plt.savefig("images/{}_{}_{}_orbital_{}_R_{}.png".format(formula,basis,model_name,nu,round(R_star*100)))
        plt.close()
        

    # plot predicted density
    vmax = 0.4
    imshow_kwargs = {
        'vmax': vmax,
        'vmin': 0,
        'cmap': 'Greys',
        'extent': (xmin, xmax, ymin, ymax),
    }

    fig, ax = plt.subplots()
    pos = ax.imshow(values, **imshow_kwargs)
    for k,Z_k in enumerate(numbers):
        if Z_k == 6:
            color = 'white'
        elif Z_k == 7:
            color = 'b'
        elif Z_k == 8:
            color = 'r'
        plt.scatter([positions[k,0]],[0],s=100,c=color)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.tick_params(bottom=False,left=False)
    #fig.colorbar(pos)
    plt.savefig("images/{}_{}_{}_density_R_{}.png".format(formula,basis,model_name,round(R_star*100)))



'''
Miscaleanous
'''

def RHF_dissociation(formula,basis,numbers,positions):
    if formula == "CO":
        N = 14
        atm = gto.M(atom = [[8,[0,0,0]]],
                basis = basis,
                unit='Angstrom')
        B = 2*atm.nao_nr()
        mf1 = scf.RHF(atm)
        mf1.kernel()

        atm = gto.M(atom = [[6,[0,0,0]]],
                        basis = basis,
                        unit='Angstrom')
        mf2 = scf.RHF(atm)
        mf2.kernel()

        E_dissoc = mf1.e_tot + mf2.e_tot


        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        N = mol.nelectron
        
        # construct Loewdin-orthonormalized Fock matrix
        S0 = mol.intor('int1e_ovlp')
        S_inv_root = scipy.linalg.sqrtm(np.linalg.inv(S0))
        F0 = torch.tensor(S_inv_root @ mf.get_fock() @ S_inv_root).to(dtype=torch.float)

        # separate energies of occupied and virtual orbitals: be careful about degenerate eigenstates, because they might be wrongly separated
        e, U = torch.linalg.eigh(F0)
        e[N//2:] += 1
        F0 = U @ torch.diag_embed(e) @ U.transpose(0,1)
        
    elif formula == "N2":
        N = 14
        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        N = mol.nelectron
        
        # construct Loewdin-orthonormalized Fock matrix
        S0 = mol.intor('int1e_ovlp')
        S_inv_root = scipy.linalg.sqrtm(np.linalg.inv(S0))
        F0 = S_inv_root @ mf.get_fock() @ S_inv_root
        E_dissoc = mf.e_tot
    
        # separate energies of occupied and virtual orbitals: be careful about degenerate eigenstates, because they might be wrongly separated
        e, U = np.linalg.eigh(F0)
        e[N//2:] += 1
        F0 = torch.tensor(U @ np.diag(e) @ U.transpose()).to(dtype=torch.float)
    else:
        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        N = mol.nelectron

        # construct Loewdin-orthonormalized Fock matrix
        S0 = mol.intor('int1e_ovlp')
        S_inv_root = scipy.linalg.sqrtm(np.linalg.inv(S0))
        F0 = torch.tensor(S_inv_root @ mf.get_fock() @ S_inv_root).to(dtype=torch.float)

        # separate energies of occupied and virtual orbitals: be careful about degenerate eigenstates, because they might be wrongly separated
        e, U = torch.linalg.eigh(F0)
        e[N//2:] += 1
        F0 = U @ torch.diag_embed(e) @ U.transpose(0,1)

    return F0, mf.e_tot


def get_basis_size(basis,numbers):
    mol = gto.M(atom = [[numbers[l],np.array([l,0,0])] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
    
    B_k = [None]*9
    for Z in [1,6,7,8,9]:
        atm = gto.M(atom = [[Z,[0,0,0]]],
                basis = basis,
                spin = spin[Z-1],
                unit='Angstrom')
        B_k[Z-1] = atm.nao_nr()
    return mol.nao_nr(), B_k, mol.nelectron, mol.natm


def get_mask(basis,B):
    if basis == 'sto-6g':
        orbitals = [0,0,1]
    elif basis == '631g':
        orbitals = [0,0,0,1,1]
    elif basis == '631g*':
        orbitals = [0,0,0,1,1,2]
    elif basis == '6311+g*':
        orbitals = [0,0,0,0,0,1,1,1,1,2]
    elif basis == 'def2-tzvp':
        orbitals = [0,0,0,0,0,1,1,1,2,2,3]
    else:
        orbitals = None

    beta = 0.5 # just something non-trivial
    R = torch.tensor([np.cos(beta)*np.sin(beta),np.sin(beta)**2,np.cos(beta)])
    Y1 = torch.tensor([sph_harm_real(R,l,m) for l in orbitals for m in range(-l,l+1)])
    Y2 = torch.tensor([sph_harm_real(-R,l,m) for l in orbitals for m in range(-l,l+1)])
    
    mask = torch.zeros(B,B,dtype=torch.double)
    mask[:B//2,:B//2] = torch.einsum('p,q->pq',Y1,Y1)
    mask[:B//2,B//2:] = torch.einsum('p,q->pq',Y1,Y2)
    mask[B//2:,:B//2] = torch.einsum('p,q->pq',Y2,Y1)
    mask[B//2:,B//2:] = torch.einsum('p,q->pq',Y2,Y2)
    mask = mask/torch.abs(mask)
    return mask

def get_h_and_S(basis,numbers,position):
    mol = gto.M(atom = [[numbers[l],position[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
    h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    S_i = mol.intor('int1e_ovlp')
    return torch.tensor(h_i), torch.tensor(S_i)

def compute_h_matrix(basis,numbers,positions,device):
    batch_size = positions.shape[0]
    out = [get_h_and_S(basis,numbers,positions[n]) for n in range(batch_size)]
    h = torch.stack([out[n][0] for n in range(batch_size)]).to(device)
    S = torch.stack([out[n][1] for n in range(batch_size)]).to(device)
    s, L = torch.linalg.eigh(S)
    S_inv_root = L @ torch.diag_embed(torch.sqrt(1/s)) @ L.transpose(1,2)
    h_new = torch.einsum('npr,nrq->npq',S_inv_root,torch.einsum('nrs,nsq->nrq', h, S_inv_root))
    return h_new #.to(dtype=torch.float)

def number_of_parameters(model):
    n_params = 0
    for param in model.parameters():
        n_params += int(torch.prod(torch.tensor(param.shape)))
    return n_params

def get_GTO_coeff(basis,numbers):
    if basis == 'sto-6g':
        orbitals = [None,[0],None,None,None,None,[0,0,1],[0,0,1],[0,0,1]]
        # alpha and d parameters of GTOs
        basis_func = [None,None,None,None,None,None,
                      [(np.array([742.7370491,136.1800249,38.09826352,13.08778177,5.082368648,2.093200076]),
                       np.array([0.009163596281,0.04936149294,0.1685383049,0.3705627997,0.4164915298,0.1303340841])),
                      (np.array([30.4972395,6.036199601,1.876046337,0.721782647,0.3134706954,0.143686555]),
                       np.array([-0.01325278809,-0.04699171014,-0.03378537151,0.2502417861,0.5951172526,0.2407061763])),
                      (np.array([30.4972395,6.036199601,1.876046337,0.721782647,0.3134706954,0.143686555]),
                       np.array([0.003759696623,0.03767936984,0.1738967435,0.4180364347,0.1017082955]))],

                      [(np.array([1027.828458,188.4512226,52.72186097,18.11138217,7.033179691,2.896651794]),
                       np.array([0.009163596281,0.0493614929,0.1685383049,0.3705627997,0.4164915298,0.1303340841])),
                      (np.array([39.19880787,7.758467071,2.411325783,0.9277239437,0.4029111410,0.1846836552]),
                       np.array([-0.01325278809,-0.04699171014,-0.03378537151,0.2502417861,0.5951172526,0.2407061763])),
                      (np.array([39.19880787,7.758467071,2.411325783,0.9277239437,0.4029111410,0.1846836552]),
                       np.array([0.003759696623,0.03767936984,0.1738967435,0.4180364347,0.4258595477,0.1017082955]))],
                       
                      [(np.array([1355.584234,248.5448855,69.53390229,23.88677211,9.275932609,3.820341298]),
                       np.array([0.009163596281,0.0493614929,0.1685383049,0.3705627997,0.4164915298,0.1303340841])),
                      (np.array([52.18776196,10.32932006,3.210344977,1.235135428,0.5364201581,0.245880606]),
                       np.array([-0.01325278809,-0.04699171014,-0.03378537151,0.2502417861,0.5951172526,0.2407061763])),
                      (np.array([52.18776196,10.32932006,3.210344977,1.235135428,0.5364201581,0.245880606]),
                       np.array([0.003759696623,0.03767936984,0.1738967435,0.4180364347,0.4258595477,0.1017082955]))]]
    elif basis == '631g':
        orbitals = [None,[0,0],None,None,None,None,[0,0,0,1,1],[0,0,0,1,1],[0,0,0,1,1]]
        basis_func = [None,
                      
                      [(np.array([18.73113696,2.825394365,0.6401216923]),
                       np.array([0.03349460434,0.2347269535,0.8137573261])),
                       (np.array([0.1612777588]),
                       np.array([1]))],
                      
                      None,None,None,None,
                      [(np.array([3047.52488,457.369518,103.94868685,29.2101553,9.28666296,3.16392696]),
                       np.array([0.001834737132,0.01403732281,0.06884262226,0.2321844432,0.4679413484,0.3623119853])),
                      (np.array([7.86827235,1.88128854,0.544249258]),
                       np.array([-0.1193324198,-0.1608541517,1.143456438])),
                      (np.array([0.1687144782]),
                       np.array([1])),
                      (np.array([7.86827235,1.88128854,0.544249258]),
                       np.array([0.06899906659,0.316423961,0.7443082909])),
                      (np.array([0.1687144782]),
                       np.array([1]))],

                      [(np.array([4173.51146,627.457911,142.902093,40.2343293,12.8202129,4.39043701]),
                       np.array([0.00183477216,0.013994627,0.06858655181,0.232240873,0.4690699481,0.3604551991])),
                      (np.array([11.62636186,2.716279807,0.7722183966]),
                       np.array([-0.1149611817,-0.1691174786,1.145851947])),
                      (np.array([0.2120314975]),
                       np.array([1])),
                      (np.array([11.62636186,2.716279807,0.7722183966]),
                       np.array([0.06757974388,0.3239072959,0.7408951398])),
                      (np.array([0.2120314975]),
                       np.array([1]))],
                       
                      [(np.array([5484.67166,825.234946,188.046958,52.9645,16.8975704,5.7996353]),
                       np.array([0.00183107443,0.0139501722,0.0684450781,0.232714336,0.470192898,0.358520853])),
                      (np.array([15.53961625,3.599933586,1.01376175]),
                       np.array([-0.1107775495,-0.1480262627,1.130767015])),
                      (np.array([0.2700058226]),
                       np.array([1])),
                      (np.array([15.53961625,3.599933586,1.01376175]),
                       np.array([0.07087426823,0.3397528391,0.7271585773])),
                      (np.array([0.2700058226]),
                       np.array([1]))]]
    elif basis == '6311+g*':
        orbitals = [None,None,None,None,None,None,None,[0,0,0,0,0,1,1,1,1,2],None]
        basis_func = [None,None,None,None,None,None,None,
                      
                      [(np.array([6293.48,949.044,218.776,63.6916,18.8282,2.72023]),
                       np.array([0.00196979,0.0149613,0.0735006,0.248937,0.60246,0.256202])),
                      (np.array([30.6331,7.02614,2.11205]),
                       np.array([0.111906,0.921666,-0.00256919])),
                      (np.array([0.684009]),
                       np.array([1])),
                      (np.array([0.200878]),
                       np.array([1])),
                      (np.array([0.0639]),
                       np.array([1])),
                      (np.array([30.6331,7.02614,2.11205]),
                       np.array([0.0383119,0.237403,0.817592])),
                      (np.array([0.684009]),
                       np.array([1])),
                      (np.array([0.200878]),
                       np.array([1])),
                      (np.array([0.0639]),
                       np.array([1])),
                      (np.array([0.913]),
                       np.array([1]))],
                       
                       None]
    elif basis == 'def2-svp':
        orbitals = [None,[0,0,1],None,None,None,None,[0,0,0,1,1,2],[0,0,0,1,1,2],[0,0,0,1,1,2]]
        basis_func = [None,[(np.array([13.0107010,1.9622572,0.44453796]),
                             np.array([0.019682158,0.13796524,0.47831935])),
                            (np.array([0.12194962]),
                             np.array([1])),
                            (np.array([0.8]),
                             np.array([1]))],None,None,None,None,
                       [(np.array([1238.4016938,186.29004992,42.251176346,11.676557932,3.5930506482]),
                         np.array([0.005456883208,0.0406384093211,0.18025593888,0.463151211755,0.44087173314])),
                        (np.array([0.40245147363]),
                         np.array([1])),
                        (np.array([0.13090182668]),
                         np.array([1])),
                        (np.array([9.4680970621,2.0103545142,0.54771004707]),
                         np.array([0.038387871728,0.21117025112,0.51328172114])),
                        (np.array([0.15268613795]),
                         np.array([1])),
                        (np.array([0.8]),
                         np.array([1]))],
                       [],
                       [(np.array([2266.1767785,340.87010191,77.363135167,21.47964494,6.6589433124]),
                         np.array([0.00534318099,0.03989003923,0.17853911985,0.4627684959,0.44309745172])),
                        (np.array([0.80975975668]),
                         np.array([1])),
                        (np.array([0.25530772234]),
                         np.array([1])),
                        (np.array([17.721504317,3.863550544,1.0480920883]),
                         np.array([0.043394573193,0.23094120765,0.51375311064])),
                        (np.array([0.27641544411]),
                         np.array([1])),
                        (np.array([1.2]),
                         np.array([1]))]]
    elif basis == 'def2-tzvp':
        orbitals = [None,[0,0],None,None,None,None,[0,0,0,0,0,1,1,1,2,2,3],[0,0,0,0,0,1,1,1,2,2,3],[0,0,0,0,0,1,1,1,2,2,3]]
        basis_func = [None,None,None,None,None,None,
                      [(np.array([13575.349682,2032.233368,463.22562359,131.20019598,42.853015891,15.584185766]),
                        np.array([0.00022245814352,0.001723273825,0.0089255715314,0.035727984502,0.11076259931,0.24295627626])),
                       (np.array([6.2067138508,2.5764896527]),
                        np.array([0.41440263448,0.23744968655])),
                       (np.array([0.57696339419]),
                        np.array([1])),
                       (np.array([0.22972831358]),
                        np.array([1])),
                       (np.array([0.095164440028]),
                        np.array([1])),
                       (np.array([34.697232244,7.9582622826,2.3780826883,0.81433208183]),
                        np.array([0.0053333657805,0.035864109092,0.14215873329,0.34270471845])),
                       (np.array([0.28887547253]),
                        np.array([1])),
                       (np.array([0.10056823671]),
                        np.array([1])),
                       (np.array([1.097]),
                        np.array([1])),
                       (np.array([0.318]),
                        np.array([1])),
                       (np.array([0.761]),
                        np.array([1]))],

                      [(np.array([19730.800647,2957.8958745,673.22133595,190.68249494,62.295441898,22.654161182]),
                        np.array([0.0002188798499,0.0016960708803,0.0087954603538,0.035359382605,0.11095789217,0.24982972552])),
                       (np.array([8.9791477428,3.686300237]),
                        np.array([0.40623896148,0.24338217176])),
                       (np.array([0.84660076805]),
                        np.array([1])),
                       (np.array([0.33647133771]),
                        np.array([1])),
                       (np.array([0.13647653675]),
                        np.array([1])),
                       (np.array([49.20038051,11.346790537,3.4273972411,1.1785525134]),
                        np.array([0.0055552416751,0.038052379723,0.14953671029,0.3494930523])),
                       (np.array([0.41642204972]),
                        np.array([1])),
                       (np.array([0.14260826011]),
                        np.array([1])),
                       (np.array([1.654]),
                        np.array([1])),
                       (np.array([0.469]),
                        np.array([1])),
                       (np.array([1.093]),
                        np.array([1]))],
                      
                      [(np.array([27032.382631,4052.3871392,922.3272271,261.24070989,85.354641351,31.035035245]),
                        np.array([0.00021726302465,0.0016838662199,0.0087395616265,0.035239968808,0.11153519115,0.25588953961])),
                       (np.array([12.260860728,4.9987076005]),
                        np.array([0.39768730901,0.2462784943])),
                       (np.array([1.1703108158]),
                        np.array([1])),
                       (np.array([0.46474740994]),
                        np.array([1])),
                       (np.array([0.18504536357]),
                        np.array([1])),
                       (np.array([63.274954801,14.627049379,4.4501223456,1.5275799647]),
                        np.array([0.0060685103418,0.041912575824,0.16153841088,0.35706951311])),
                       (np.array([0.52935117943]),
                        np.array([1])),
                       (np.array([0.1747842127]),
                        np.array([1])),
                       (np.array([2.314]),
                        np.array([1])),
                       (np.array([0.645]),
                        np.array([1])),
                       (np.array([1.428]),
                        np.array([1]))]]
    else:
        orbitals = None
        
    for Z in numbers:
        for mu, (a,d) in enumerate(basis_func[Z]):
            l = orbitals[Z][mu]
            # norm = np.sum([d[i] * d[j] * spherical_integral_1D(a[i],a[j],l) for i in range(len(a)) for j in range(len(a)) ])
            norm = 1/np.sqrt(contracted_integral_norm(l,0,l,0,0,0,0,0,
                                a, a, [0], [0], d, d, [1], [1],
                                np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])))
            # norm = np.sum([d[i] * d[j] *\
            #        spherical_integral(l,0,l,0,0,0,0,0,a[i],a[j],0,0,
            #                           np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])) *GTO_norm(a[i],l)*GTO_norm(a[j],l)
            #        for i in range(len(a)) for j in range(len(a)) ])
            
            for j in range(len(d)):
                basis_func[Z][mu][1][j] = norm*d[j]
    
    return basis_func, orbitals

def test_diatomics(model,basis,numbers,increments,indices):
    """
    Evaluates the model on 200 equidistant points between 0.7 and 2 Angstrom, computes the MAE and plots the absolute error in dependence of
    the atomic distance.
    """
    plot_lo = 0.7
    plot_hi = 2
    plot_increments = 200
    n_atoms = 2

    B, B_k, N, M = get_basis_size(basis,numbers)
    positions = np.zeros((plot_increments,n_atoms,3))
    h_s = np.zeros((plot_increments,B,B))
    energies = np.zeros(plot_increments)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)):
        position = np.array([[-R/2,0,0],[R/2,0,0]])
        positions[i] = position

        mol = gto.M(atom = [[numbers[l],position[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        energies[i] = mf.e_tot

        S = mol.intor('int1e_ovlp')
        h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_inv_root = np.linalg.inv(scipy.linalg.sqrtm(S)).real
        h_s[i] = S_inv_root @ h @ S_inv_root

    positions = torch.tensor(positions,dtype=torch.float).reshape((plot_increments,n_atoms,3))
    h_s = torch.tensor(h_s,dtype=torch.double)
    F_new, e, E = model((positions,h_s))
    
    # compute absolute error
    E_pred = E.detach().numpy()*Hartree_to_eV/Hartree_to_kcal_per_mol
    E_ref = energies*Hartree_to_eV/Hartree_to_kcal_per_mol
    AE = E_pred-E_ref
    MAE = np.mean(np.abs(AE))
    print("MAE:",MAE,"kcal/mol")

    # plot absolute values of the energy
    E_min = np.min(E_ref)
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),E_ref-E_min,label="reference",color="k")
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),E_pred-E_min,label="prediction",color="r")
    plt.xlabel("atomic distance in Angstrom")
    plt.ylabel(r"absolute error $|E_{\text{pred}}-E_{\text{ref}}|$ in kcal/mol")
    plt.legend()
    plt.show()

    # plot energy difference
    plt.plot(np.linspace(0.7,2,increments)[indices],[0]*90,"o",color="k",label="training points")
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),AE,label="prediction error")
    plt.xlabel("atomic distance in Angstrom")
    plt.ylabel(r"prediction error $E_{\text{pred}}-E_{\text{ref}}$ in kcal/mol")
    plt.legend()
    plt.show()
    return
