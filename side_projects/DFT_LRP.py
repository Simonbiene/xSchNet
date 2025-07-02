import numpy as np
from pyscf import scf, gto
from ase.db import connect


def nuc(k):
    mol = gto.M(atom = [[numbers[l],positions[l]-positions[k]] for l in range(M)],
                basis = basis,
                unit='Angstrom')
    nuc_k = mol.intor('int1e_rinv')
    return nuc_k


data = connect('qm9.db')
for i in [3]:
    if i == 0:
        R = 0.74
        numbers = np.array([1,1])
        positions = [[0,0,0],[R,0,0]]
        positions = list(map(np.array,positions))
    else:
        for row in data.select(i):
            formula = row.formula
            print(formula)
            print(row.data["energy_U0"][0],"a.u.")
            numbers, positions = row.numbers, row.positions

#R_Z = np.array([6,7,1])
#R_Z = np.array([-0.613,-0.613])
#R_Z = np.array([-38.08,-0.6,-0.6,-0.6,-0.6]) # methane
R_Z = np.array([-75.23,-0.6,-0.6]) # water 
#R_Z = np.array([-38.055,-38.055,-0.6,-0.6])
#R_Z = np.array([-38.1,-54.71,-0.6])
basis = 'sto-6g'

mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
N = mol.nelectron
B = mol.nao_nr()
M = mol.natm

mf = scf.RHF(mol)
mf.kernel()
e = mf.mo_energy[:N//2]
U = mf.mo_coeff[:,:N//2]
h = np.zeros((B*N//2,B*N//2))
h_nuc = np.zeros((B*N//2,B*N//2))
S = np.zeros((B*N//2,B*N//2))
S_inv = np.zeros((B*N//2,B*N//2))
J = np.zeros((B*N//2,B*N//2))
K = np.zeros((B*N//2,B*N//2))
S_i = mol.intor('int1e_ovlp')
h_i = mol.intor('int1e_kin')
v = mol.intor('int2e')
J_i = np.einsum('pqii->pq',np.einsum('pqrj,ri->pqij',np.einsum('pqrs,sj->pqrj',v,U), U))
K_i = np.einsum('piqi->pq',np.einsum('prqj,ri->piqj',np.einsum('prqs,sj->prqj',v,U), U))
n_k = 0
for i in range(N//2):
    S[n_k:n_k+B,n_k:n_k+B] = e[i]*S_i
    S_inv[n_k:n_k+B,n_k:n_k+B] = np.linalg.inv(S_i)
    h[n_k:n_k+B,n_k:n_k+B] = h_i
    h_nuc[n_k:n_k+B,n_k:n_k+B] = mol.intor('int1e_nuc')
    J[n_k:n_k+B,n_k:n_k+B] = J_i
    K[n_k:n_k+B,n_k:n_k+B] = K_i
    n_k += B

h_nuc_k = np.stack([nuc(k) for k in range(M)])
A = np.einsum('kpq,qi->ipk',h_nuc_k,U).reshape((B*N//2,M))

c = U.transpose().reshape(B*N//2)
W = np.linalg.inv(A.transpose() @ S_inv @ A) @ A.transpose() @ S_inv @ (h + 2*J - K - S) # shape (M,B*N//2)

Z = W @ c
print(Z)

R_c = c * ((W / (W @ c).reshape((M,1))).transpose() @ R_Z)

X = (W / (W @ c).reshape((M,1))).transpose()
print("necessary relevances for a quantity, where s-orbital of oxygen doesn't contribute (e.g. kinetic energy):")
print("hydrogen relevance:",X[0,0]*mf.e_tot/2/(X[0,0]-X[0,1]))
print("oxygen relevance:",-2*X[0,1]*mf.e_tot/2/(X[0,0]-X[0,1]))
R_Z = [-2*X[0,1]*mf.e_tot/2/(X[0,0]-X[0,1]), X[0,0]*mf.e_tot/2/(X[0,0]-X[0,1]), X[0,0]*mf.e_tot/2/(X[0,0]-X[0,1])]
R_c = c * ((W / (W @ c).reshape((M,1))).transpose() @ R_Z)


float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(R_c.reshape((N//2,B)).transpose())
#print(np.sum(R_c.reshape((N//2,B)),axis=1))
print(e + np.diag(U.transpose() @ (h_i + mol.intor('int1e_nuc')) @ U ))