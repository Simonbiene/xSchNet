from pyscf import gto, scf

import torch
import os
import numpy as np

from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.integrate import quad, dblquad

import schnetpack as spk

from ase import Atoms
from ase.db import connect
from utils.utils import build_atomwise_model
from time import time

atomrefs = [0,-13.6131,0,0,0,0,-1029.863,-1485.3025,-2042.6111,-2713.4849]
mean = -4.2427 # eV
spin = [1,0,1,0,1,2,3,2,1,0]

pople_6311ppgss_radial_basis_l = [[4,1],None,None,None,None,[5,4,1],[5,4,1],[5,4,1]]
pople_6311ppgss_angular_momentum = [2,None,None,None,None,3,3,3]
pople_6311ppgss = [[([33.86500,5.094790,1.158790],[0.0254938,0.190373,0.852161],0),
                   ([0.325840],[1],0),
                   ([0.102741],[1],0),
                   ([0.0360000],[1],0),
                   ([0.750],[1],1)],
                   None,
                   None,
                   None,
                   None,
                   [([4563.240,682.0240,154.9730,44.45530,13.02900,1.827730],[0.00196665,0.0152306,0.0761269,0.2608010,0.6164620,0.2210060],0),
                    ([20.96420,4.803310,1.459330],[0.114660,0.919999,-0.00303068],0),
                    ([0.4834560],[1],0),
                    ([0.1455850],[1],0),
                    ([0.0438000],[1],0),
                    ([20.96420,4.803310,1.459330],[0.0402487,0.237594,0.815854],1),
                    ([0.4834560],[1],1),
                    ([0.1455850],[1],1),
                    ([0.0438000],[1],1),
                    ([0.626],[1],2)],
                   [([6293.480,949.0440,218.7760,63.69160,18.82820,2.720230],[0.00196979,0.0149613,0.0735006,0.2489370,0.6024600,0.2562020],0),
                    ([30.63310,7.026140,2.112050],[0.111906,0.921666,-0.00256919],0),
                    ([0.684009],[1],0),
                    ([0.200878],[1],0),
                    ([0.0639000],[1],0),
                    ([30.63310,7.026140,2.112050],[0.0383119,0.237403,0.817592],1),
                    ([0.684009],[1],1),
                    ([0.200878],[1],1),
                    ([0.0639000],[1],1),
                    ([0.913],[1],2)],
                   [([8588.500,1297.230,299.2960,87.37710,25.67890,3.740040],[0.00189515,0.0143859,0.0707320,0.2400010,0.5947970,0.2808020],0),
                    ([42.11750,9.628370,2.853320],[0.113889,0.920811,-0.00327447],0),
                    ([0.905661],[1],0),
                    ([0.255611],[1],0),
                    ([0.0845000],[1],0),
                    ([42.11750,9.628370,2.853320],[0.0365114,0.237153,0.819702],1),
                    ([0.905661],[1],1),
                    ([0.255611],[1],1),
                    ([0.0845000],[1],1),
                    ([1.292],[1],2)]]

def import_data(dataset_size):
     data = connect('code/qm9.db')
     molecules = []

     if dataset_size == 'all':
          selected_data = data.select()
          for row in selected_data:
               molecules.append([row.numbers, row.positions])
     else:
          for i in list(range(dataset_size)):
               for row in data.select(i):
                    molecules.append([row.numbers, row.positions])
     return molecules

def import_best_model():
     qm9tut = './code/qm9tut'
     best_model = torch.load(os.path.join(qm9tut, 'qm9_new_schnet_5_ssp/best_model'), map_location="cpu")
     return best_model

def create_SchNet_predictions(molecule):
     # rebuild model atom-wise
     atomwise_model = build_atomwise_model(best_model)
     A = len(molecule[0])
     numbers = molecule[0]
     positions = molecule[1]
     converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=5.)) # dtype=torch.float32
     with torch.no_grad():
        atomic_energies = np.zeros(A)
        neighbor_indices = [None]*A
        neighbor_distances = [None]*A
        for k in range(A):
            atomic_energies[k] = atomrefs[molecule[0][k]]
            distances = np.array([np.linalg.norm(positions[k]-positions[l]) for l in range(len(positions))])
            neighbor_indices[k] = np.nonzero([(0 < distances) & (distances < 5)])[1]

        atoms = Atoms(numbers=numbers, positions=positions)
        inputs = converter(atoms)
        E_pred_k = atomwise_model(inputs)["energy_U0"] # in eV
        E = np.array(E_pred_k + atomic_energies + mean)

     return E, numbers, positions, neighbor_indices, distances

def get_initial_guesses_and_matrices(basis):
     list_of_S = [None]*8
     list_of_h = [None]*8
     list_of_v = [None]*8
     list_of_inital_guesses = [None]*8
     list_of_n_basis = [None]*8
     for Z in [1,6,7,8]:
          mol = gto.M(atom = [[Z,[0,0,0]]],
                      basis = basis,
                      spin=spin[Z-1]
                      )

          ovlp = mol.intor('int1e_ovlp')
          kin = mol.intor('int1e_kin')
          nuc = mol.intor('int1e_nuc')
          v = mol.intor('int2e')
          uhf = scf.UHF(mol)
          uhf.kernel()
          U_a, U_b = uhf.mo_coeff

          list_of_S[Z-1] = ovlp
          list_of_h[Z-1] = kin+nuc
          list_of_v[Z-1] = v
          list_of_inital_guesses[Z-1] = (np.transpose(U_a), np.transpose(U_b))
          list_of_n_basis[Z-1] = mol.nao_nr()
     return list_of_S, list_of_h, list_of_v, list_of_inital_guesses, list_of_n_basis

def half_factorial(n):
    p=1
    for i in range(0,n+1,2):
        p *= max(n-i,1)
    return p

def R(r,a,l):
    '''
    primitve GTO radial function with exponent a and quantum number l
    '''
    return 2*(2*a)**0.75/np.pi**0.25*np.sqrt(2**l/half_factorial(2*l+1))*(np.sqrt(2*a)*r)**l*np.exp(-a*r**2)

def contracted_gaussian(r,a,d,l):
     return sum([d[i]*R(r,a[i],l) for i in range(len(a))])

def normierung(r,i,basis_parameters):
    a=basis_parameters[i][0]
    d=basis_parameters[i][1]
    p=basis_parameters[i][2]
    s1 = 0
    for k in range(len(a)):
        s1 += d[k]*R(r,a[k],p)
    return s1**2*r**2

def integrand_n(r,i,j,basis_parameters,neighbor_indices_k):
    ai=basis_parameters[i][0]
    di=basis_parameters[i][1]
    li=basis_parameters[i][2]
    aj=basis_parameters[j][0]
    dj=basis_parameters[j][1]
    lj=basis_parameters[i][2]
    
    # sum over attracting potentials
    s = -sum([numbers[l]/max(r,distances[l]) for l in neighbor_indices_k])
    return contracted_gaussian(r,ai,di,li)*contracted_gaussian(r,aj,dj,lj)*s*r**2

def compute_nuclear_attraction_matrix(Z,neighbor_indices_k):
     '''
     Warning: Integrals are potentially divergent
     '''
     n_basis = list_of_n_basis[Z-1]
     nuclear_attraction = np.zeros((n_basis,n_basis))
     if basis == '6311++g**':
          basis_parameters = pople_6311ppgss[Z-1]
          l_max = pople_6311ppgss_angular_momentum[Z-1]
          n = pople_6311ppgss_radial_basis_l[Z-1]
          n_radial_basis = sum(n)

     # normalizing factor for basis functions
     N = np.zeros(n_radial_basis) 
     for i in range(n_radial_basis):
          N[i] = np.sqrt(quad(lambda r: normierung(r,i,basis_parameters),0,np.inf)[0])
     
     #neighbor_distances = [0] + np.sort(np.take(distances,neighbor_indices_k)) + [np.inf]
     for l in range(l_max):
          mu = sum([n[p] for p in range(l)])
          for i in range(n[l]):
               for m in range(-l,l+1):
                    # diagonal element
                    nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
                    nuclear_attraction[nu_i,nu_i] = 1/N[i+mu]**2*quad(lambda r: integrand_n(r,i+mu,i+mu,basis_parameters,neighbor_indices_k),0,np.inf)[0]
                    #for a in range(len(neighbor_distances)-1):
                         #nuclear_attraction[nu_i,nu_i] += quad(lambda r: integrand_n(r,i+mu,i+mu,basis_parameters,neighbor_indices_k),neighbor_distances[a],neighbor_distances[a+1])[0]
                    
                    # off-diagonal elements
                    for j in range(i):
                         nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
                         nu_j = m+l + (2*l+1)*j + sum([(2*p+1)*n[p] for p in range(l)])
                         
                         nuclear_attraction[nu_i,nu_j] = 1/N[i+mu]/N[j+mu]*quad(lambda r: integrand_n(r,i+mu,j+mu,basis_parameters,neighbor_indices_k),0,np.inf)[0]
                         nuclear_attraction[nu_j,nu_i] = nuclear_attraction[nu_i,nu_j]

     return nuclear_attraction

def symmetrized_density(r,Z,U_a,U_b,normalization):
     N_a = int((Z-spin[Z-1])//2 + spin[Z-1])
     N_b = int((Z-spin[Z-1])//2)
     if basis == '6311++g**':
          basis_parameters = pople_6311ppgss[Z-1]
          l_max = pople_6311ppgss_angular_momentum[Z-1]
          n = pople_6311ppgss_radial_basis_l[Z-1]
     
     s = 0
     for l in range(l_max):
          mu = sum([n[p] for p in range(l)])
          for i in range(n[l]):
               for m in range(-l,l+1):
                    # diagonal element
                    nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])

                    # off-diagonal elements
                    for j in range(i):
                         nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
                         nu_j = m+l + (2*l+1)*j + sum([(2*p+1)*n[p] for p in range(l)])
                         ai=basis_parameters[i+mu][0]
                         di=basis_parameters[i+mu][1]
                         aj=basis_parameters[j+mu][0]
                         dj=basis_parameters[j+mu][1]
                         Da_ij = sum([U_a[nu_i,nu]*U_a[nu_j,nu] for nu in range(N_a)])
                         Db_ij = sum([U_b[nu_i,nu]*U_b[nu_j,nu] for nu in range(N_b)])
                         s += 2*(Da_ij + Db_ij)/normalization[Z-1][i+mu]/normalization[Z-1][j+mu]*contracted_gaussian(r,ai,di,l)*contracted_gaussian(r,aj,dj,l)
     
     return s

def integrand_e(r1,r2,i,j,basis_parameters,neighbor_indices_k,list_of_orbitals,N):
     '''
     inputs:
     -list_of_orbitals_k: list of matrices (U_a,U_b) for each neighbor of atom k, which columns are the orbital coefficients from the previous layer
     '''
     
     ai=basis_parameters[i][0]
     di=basis_parameters[i][1]
     li=basis_parameters[i][2]
     aj=basis_parameters[j][0]
     dj=basis_parameters[j][1]
     lj=basis_parameters[i][2]

     s = sum([symmetrized_density(abs(r2-distances[l]),numbers[l],list_of_orbitals[l][0],list_of_orbitals[l][1],N) for l in neighbor_indices_k])
     result = contracted_gaussian(r1,ai,di,li)*contracted_gaussian(r1,aj,dj,lj)*s*r1**2*r2**2
     if np.isnan(result):
          print(contracted_gaussian(r1,ai,di,li))
          print(contracted_gaussian(r1,aj,dj,lj))
          print(abs(r1-r2))
          print(s)
          os._exit(-1)
     if r1==r2 and result != 0:
          return 1e20
     else:
          return result/abs(r1-r2)

def compute_electronic_repulsion_matrix(Z_k,neighbor_indices_k,list_of_orbitals):
     n_basis = list_of_n_basis[Z_k-1]
     electronic_repulsion = np.zeros((n_basis,n_basis))
     if basis == '6311++g**':
          basis_parameters = pople_6311ppgss[Z_k-1]
          l_max = pople_6311ppgss_angular_momentum[Z_k-1]
          n = pople_6311ppgss_radial_basis_l[Z_k-1]
          n_radial_basis = sum(n)

     # normalizing factors for basis functions
     N = [None]*8
     for Z in [1,6,7,8]:
          parameters = pople_6311ppgss[Z-1]
          m = pople_6311ppgss_radial_basis_l[Z-1]
          m_radial_basis = sum(m)
          N_Z = np.zeros(m_radial_basis) 
          for i in range(m_radial_basis):
               N_Z[i] = np.sqrt(quad(lambda r: normierung(r,i,parameters),0,np.inf)[0])
          N[Z-1] = N_Z
     
     # precalculate sums of coefficients
     '''
     N_a = int((Z_k-spin[Z_k-1])//2 + spin[Z_k-1])
     N_b = int((Z_k-spin[Z_k-1])//2)
     U_a, U_b = list_of_orbitals[Z_k-1]
     D_a = U_a[:,:N_a] @ np.transpose(U_a[:,:N_a])
     D_b = U_b[:,:N_b] @ np.transpose(U_b[:,:N_b])
     D = D_a + D_b
     for l in range(l_max):
          for m in range(-l,l+1):
               nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
     '''
     N_k = N[Z_k-1] # normalizing constant for radial GTO
     for l in range(l_max):
          mu = sum([n[p] for p in range(l)])
          for i in range(n[l]):
               start = time()
               for m in range(-l,l+1):
                    # diagonal element
                    nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
                    electronic_repulsion[nu_i,nu_i] = 0.5/N_k[i+mu]**2*dblquad(lambda r1,r2: integrand_e(r1,r2,i+mu,i+mu,basis_parameters,neighbor_indices_k,list_of_orbitals,N),0,10,0,np.inf)[0]
                    print(electronic_repulsion[nu_i,nu_i])
                    # off-diagonal elements
                    for j in range(i):
                         nu_i = m+l + (2*l+1)*i + sum([(2*p+1)*n[p] for p in range(l)])
                         nu_j = m+l + (2*l+1)*j + sum([(2*p+1)*n[p] for p in range(l)])
                         J_ij = 0.5/N_k[i+mu]/N_k[j+mu]*dblquad(lambda r1,r2: integrand_e(r1,r2,i+mu,j+mu,basis_parameters,neighbor_indices_k,list_of_orbitals,N),0,10,0,np.inf)[0]
                         print(J_ij)
                         electronic_repulsion[nu_i,nu_j] = J_ij
                         electronic_repulsion[nu_j,nu_i] = J_ij
               stop = time()
               print(i+1+mu,"out of",n_radial_basis)
               print("computation time:",stop-start)

     return electronic_repulsion

def get_two_electron_operators(N_a,N_b,U_a,U_b,v):
     n = U_a.shape[0]
     J = np.zeros((n,n))
     K_a = np.zeros((n,n))
     K_b = np.zeros((n,n))
     for j in range(N_a):
        J += np.matmul(np.matmul(v,U_a[:,j]),U_a[:,j])
        K_a += np.einsum('ijl,j->il',np.einsum('ijkl,k->ijl',v,U_a[:,j]),U_a[:,j])
     for j in range(N_b):
        J += np.matmul(np.matmul(v,U_b[:,j]),U_b[:,j])
        K_b += np.einsum('ijl,j->il',np.einsum('ijkl,k->ijl',v,U_b[:,j]),U_b[:,j])

     return J, K_a, K_b

def energy_calculator(Z,V_n,V_e,U_a,U_b,lambda_k):
     '''
     calculates energy of the atom
     '''
     h = list_of_h[Z-1]
     int1e = h + V_n + lambda_k*V_e

     if Z==1:
          e, U_a = eigh(int1e)
          U_b = np.zeros(U_a.shape)
          e_tot = e[0]
     else:
        N_a = int((Z-spin[Z-1])//2 + spin[Z-1])
        N_b = int((Z-spin[Z-1])//2)
        S = list_of_S[Z-1]
        v = list_of_v[Z-1]

        last_e_tot = 0
        e_tot = 1
        while abs(e_tot - last_e_tot) > 0.00001:
            last_e_tot = e_tot
            J,K_a,K_b = get_two_electron_operators(N_a,N_b,U_a,U_b,v)
            
            # diagonalize Fock matrix
            e_a, U_a = eigh(int1e+J-K_a,S)
            e_b, U_b = eigh(int1e+J-K_b,S)
            
            # compute energy
            M_alpha = np.transpose(U_a,0,1) @ int1e @ U_a
            M_beta = np.transpose(U_b,0,1) @ int1e @ U_b
            e_tot = np.trace(M_alpha[:N_a,:N_a]) + np.sum(e_a[:N_a]) + np.trace(M_beta[:N_b,:N_b]) + np.sum(e_b[:N_b])
            
     return e_tot, U_a, U_b

def get_optimized_energies_and_orbitals(Z,E_SchNet,nuclear_attraction_matrix,electronic_repulsion_matrix,U0_a,U0_b):
     res = minimize(lambda lambda_k: abs(E_SchNet-energy_calculator(Z,nuclear_attraction_matrix,electronic_repulsion_matrix,U0_a,U0_b,lambda_k)[0]),x0=[1])
     lambda_optimized = res.x
     E, U_a, U_b = energy_calculator(Z,nuclear_attraction_matrix,electronic_repulsion_matrix,U0_a,U0_b,lambda_optimized)
     return E, U_a, U_b

def compute_XAI_orbitals_and_energies(E_SchNet,numbers,neighbor_indices):
     E = np.zeros((len(numbers),4))
     A = len(numbers)
     list_of_orbitals = [None]*A
     list_of_nuclear_attraction_matrices = [None]*8
     for k,Z in enumerate(numbers):
          list_of_orbitals[k] = list_of_inital_guesses[Z-1]
          list_of_nuclear_attraction_matrices[k] = compute_nuclear_attraction_matrix(Z,neighbor_indices[k])
     for t in range(4):
          print(t)
          for k,Z in enumerate(numbers):
               nuclear_attraction_matrix = list_of_nuclear_attraction_matrices[k]
               U0_a, U0_b = list_of_orbitals[k]

               # compute matrix of electron repulsion with neighboring atoms (takes some time)
               electronic_repulsion_matrix = compute_electronic_repulsion_matrix(Z,neighbor_indices[k],list_of_orbitals)
               print(electronic_repulsion_matrix)
               
               print(k)
               # optimize the parameter lambda (longest part)
               E_e, U_a, U_b = get_optimized_energies_and_orbitals(Z,E_SchNet[k],nuclear_attraction_matrix,electronic_repulsion_matrix,U0_a,U0_b)
               
               # add nuclear repulsion energies and save
               neighbor_numbers = np.take(numbers,neighbor_indices)
               neighbor_distances = np.take(distances,neighbor_indices)
               nuclear_repulsion = 0.5*numbers[k]*np.sum(np.divide(neighbor_numbers,neighbor_distances))
               E[k,t] = E_e + nuclear_repulsion

               # update orbitals
               list_of_orbitals[k] = (U_a, U_b)
     return


# main
basis = '6311++g**' # 'def2-tzvp'
datapool_size = 200
molecules = import_data(datapool_size)
print(molecules[0][0])
list_of_S, list_of_h, list_of_v, list_of_inital_guesses, list_of_n_basis = get_initial_guesses_and_matrices(basis)
best_model = import_best_model()
E_SchNet, numbers, positions, neighbor_indices, distances = create_SchNet_predictions(molecules[0])

E, orbital_coeff = compute_XAI_orbitals_and_energies(E_SchNet, numbers, neighbor_indices)