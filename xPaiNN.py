import numpy as np
import scipy
import torch
from torch import nn
import matplotlib.pyplot as plt
from time import time

import schnetpack as spk
from ase import Atoms
from pyscf import gto, scf

from utils.utils import *
from utils.atoms import AtomsData


a0 = 0.5291772 # Angstrom

class SchNOrbInteraction(nn.Module):

    def __init__(self,n_factors,n_rbf,idx_j,n_atoms):
        super(SchNOrbInteraction, self).__init__()

        self.idx_j = idx_j
        self.n_atoms = n_atoms
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
        xj = torch.gather(x,1,self.idx_j[None,:,None,None].expand(batch_size,-1,3,self.n_factors)).reshape((batch_size,self.n_atoms,self.n_atoms-1,3,self.n_factors))
        h_ij = torch.einsum('nkdi,nkldi->nkldi', x, torch.einsum('nkli,nkldi->nkldi',W,xj) )

        # get neighbors
        nbh = neighbors[None,:,:,None,None].expand((batch_size,self.n_atoms,self.n_atoms-1,3,self.n_factors))

        p_ij_pair = self.pairnet(h_ij)
        p_ij_env = self.envnet(h_ij)
        p_i_env = torch.sum(p_ij_env,dim=2,keepdim=True)
        p_j_env = torch.gather(p_i_env.expand(-1,-1,n_atoms-1,-1,-1),1,nbh)
        p_ij = p_ij_pair + p_j_env + p_i_env

        return p_ij

class xPaiNN(nn.Module):
    """
    Args:
        basis: name of basis set for orbital expansion
        F_offset: offset Fock matrix
        n_interactions: number of PaiNN layers
        n_atom_basis: feature dimension in PaiNN interaction steps
        cut: cutoff radius
        radial_basis: type of radial basis function
        cutoff_fn: type of cutoff function
        lmax: parameter for number of SchNOrb layers (maximal angular momentum)
        n_SchNOrb: feature dimension in SchNOrb interaction steps
        fixed: whether to use h and nuc as fixed or depending on the atom positions- default False
        delta_E: generic offset between HF energy and energy at the level of theory from the reference - default 0 (if reference values are HF energies)
    """
    def __init__(self,
                 basis,F_offset,
                 n_interactions=2,
                 n_atom_basis=128,
                 cut=10,
                 radial_basis='Gaussian',
                 cutoff_fn='cosine',
                 n_SchNOrb=128,
                 fixed=False,
                 delta_E=0):
        super(xPaiNN, self).__init__()
        
        B, B_k, N, n_atoms = get_basis_size(basis,numbers)

        self.B = B
        self.B_k = B_k
        self.Bmax = max([B_k[Z_k-1] for Z_k in numbers])
        self.n_atom_basis = n_atom_basis
        self.h_is_available = False
        self.register_buffer('delta_E', delta_E)
        self.delta_E = delta_E
        if fixed:
            self.register_buffer('h0_new', h0_new) # self.h0_new = h0_new # 
            self.register_buffer('nuc0', nuc0) # self.nuc0 = nuc0 # 

        self.register_buffer('F_offset', F_offset)
        self.converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=cut),device=device)

        if formula=="N2" or formula=="CO" or formula=="O2":
            self.mask = get_mask(basis,B)
        else:
            self.mask = torch.ones(B,B).to(device)

        if radial_basis=='Gaussian':
            self.radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cut)
        elif radial_basis=='Bessel':
            self.radial_basis = spk.nn.BesselRBF(n_rbf=20, cutoff=cut)

        if cutoff_fn=='cosine':
            self.cutoff_fn = spk.nn.CosineCutoff(cutoff=cut)
        elif cutoff_fn=='mollifier':
            self.cutoff_fn = spk.nn.MollifierCutoff(cutoff=cut)

        self.preprocessor = spk.atomistic.PairwiseDistances()

        # PaiNN
        self.PaiNN = spk.representation.PaiNN(n_atom_basis=n_atom_basis,
                                              n_interactions=n_interactions,
                                              radial_basis=self.radial_basis,
                                              cutoff_fn=self.cutoff_fn)

        # transfer
        self.n_SchNOrb = n_SchNOrb
        self.transfer = nn.Linear(n_atom_basis,n_SchNOrb)

        # prediction head
        self.WF_on = nn.Linear(n_SchNOrb*3,self.Bmax**2)
        self.WF_off = nn.Linear(n_SchNOrb*3,self.Bmax**2)

        return
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def h_available(self):
        self.h_is_available = True

    def PaiNN_interactions(self,position):
        # pre-processing
        input = self.converter(Atoms(numbers=numbers, positions=position))
        input = self.preprocessor(input)
        
        # PaiNN
        outputs = self.PaiNN(input)
        return outputs["vector_representation"]


    def forward(self,input):
        """
        Args:
            input: either tuple of atom positions (torch.tensor) and Löwdin-orthonormalized one-electron matrix (torch.tensor) or just atom positions
        Returns:
            F: Löwdin-orthonromalized Fock matrix
            e: orbital energies (for both occupied and virtual states)
            E: total energy
        """
        if self.h_is_available:
            positions = input[0]
            h_new = input[1]
        else:
            positions = input 
            h_new = compute_h_matrix(basis,numbers,positions,device)
        batch_size = positions.shape[0]
        
        # PaiNN interactions
        mu = torch.stack([self.PaiNN_interactions(positions[n]) for n in range(batch_size)]) 
        x = self.transfer(mu)
        
        # pairwise interactions
        p_ij = torch.einsum('nkdi,nldi->nkldi',x,x).reshape((batch_size,n_atoms,n_atoms,3*self.n_SchNOrb))
        
        # construct the electron repulsion matrix
        delta_F = torch.zeros((batch_size,self.B,self.B)).to(device)
        n_k = 0
        for k, Z_k in enumerate(numbers):
            Bk = self.B_k[Z_k-1]

            # get on-diagonal block-matrices
            F_on = torch.sum(self.WF_on(p_ij[:,k,:,:]),dim=1).reshape((batch_size,self.Bmax,self.Bmax))
            
            # symmetrize
            F_on = 0.5*(F_on + F_on.transpose(1,2))
            
            delta_F[:,n_k:n_k+Bk,n_k:n_k+Bk] = F_on[:,:Bk,:Bk]
            n_l = 0
            for l in range(k):
                Bl = self.B_k[numbers[l]-1]

                # get off-diagonal matrix
                F_off = self.WF_off(p_ij[:,k,l,:]).reshape((batch_size,self.Bmax,self.Bmax))

                # symmetrize, if atoms are indistinguishable (=have same distances to neighbors and have same type)
                # F_off = 0.5*(F_off + F_off.transpose(1,2)) + F_off*torch.mean((x[:,k]-x[:,l]),dim=(1,2)).reshape((batch_size,1,1))
                
                # fill the overall matrix
                delta_F[:,n_k:n_k+Bk,n_l:n_l+Bl] = F_off[:,:Bk,:Bl]
                delta_F[:,n_l:n_l+Bl,n_k:n_k+Bk] = F_off[:,:Bk,:Bl].transpose(1,2)
                    
                n_l += Bl

            n_k += Bk
        
        # apply mask and offset
        zeros = torch.zeros(h_new.shape)
        zeros[torch.abs(h_new)>1e-7] = 1
        F = zeros * self.mask * delta_F * 1e-4 + self.F_offset
        if F.requires_grad:
            F.retain_grad()
        
        # solve Fock equation
        e, U = torch.linalg.eigh(F)
        
        # compute energy
        if fixed:
            E_e = torch.sum(e[:,:N//2],dim=1) + torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('pq,nqj->npj', self.h0_new, U[:,:,:N//2])))
            nuc = self.nuc0
        else:
            E_e = torch.sum(e[:,:N//2],dim=1) + torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', h_new, U[:,:,:N//2])))
            nuc = torch.sum(torch.stack([numbers[k]*numbers[l]/torch.norm(positions[:,k]-positions[:,l],dim=1)*a0 
                                         for k in range(n_atoms) for l in range(k)],dim=-1),dim=-1).to(device)
        
        return F, e, E_e + nuc + self.delta_E


class SchNet_XAIDataSet(torch.utils.data.Dataset):
    def __init__(self, positions, energies, indices):
        # create input and target data
        self.indices = indices
        
        # compute h matrices
        self.positions = positions[indices] # torch.stack([torch.tensor(database.get_properties(int(str(i)))[1]["_positions"]) for i in indices]).to(dtype=torch.double)
        self.energies = energies[indices] # torch.stack([torch.tensor(database.get_properties(int(str(i)))[1]["energy"][0]) for i in indices]).to(device)
        self.h_s = compute_h_matrix(basis,numbers,self.positions,device)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, n):
        input = (self.positions[n], self.h_s[n])
        target = self.energies[n] # (self.F[n], self.S[n], self.energies[n]) # 
        return input, target


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self,output,target):
         '''
         Loss terms for:
         -energy
         -occupied states (sould be negative)
         -HOMO-LUMO gap (should be >0.1 to avoid crossings)
         '''
         F_new, e, E = output
         E_ref = target
         return torch.mean( (E_ref - E)**2 )


def train_one_epoch(model):
     running_loss = 0.
     for i, data in enumerate(training_loader):
          # Every data instance is an input + label pair
          inputs, targets = data
          
          # Zero your gradients for every batch!
          optimizer.zero_grad()
          
          # Make predictions for this batch
          outputs = model(inputs)
          
          # Compute the loss and its gradients
          energy_loss = float(torch.mean(torch.abs(outputs[-1] - targets)))
          loss = loss_fn(outputs, targets)
          loss.backward(retain_graph=True)

          # if there are points with eigenvalue degeneracies, remove them from the gradient computation
          if torch.sum(torch.isnan(outputs[0].grad))>0:
              print("eigenvalue error\n")
              error_batches = torch.isnan(torch.sum(outputs[0].grad,dim=(1,2)))

              inputs = (inputs[0][~error_batches],inputs[1][~error_batches,:,:],inputs[2][~error_batches,:,:])
              optimizer.zero_grad()
              outputs = model(inputs)
              targets = targets[~error_batches]
              loss = loss_fn(outputs, targets)
              loss.backward(retain_graph=True)
          
          # Adjust learning weights
          optimizer.step()
          
          # Gather data and report
          running_loss += energy_loss

     return running_loss / (i + 1)


def train(model,save=True):
     best_loss = 1_000_000.

     for epoch in range(EPOCHS):
          print('EPOCH {}:'.format(epoch + 1))
          start=time()
          # Make sure gradient tracking is on, and do a pass over the data
          model.train(True)
          avg_loss = train_one_epoch(model)

          running_vloss = 0.0
          # Set the model to evaluation mode
          model.eval()

          # Disable gradient computation and reduce memory consumption.
          with torch.no_grad():
               for i, vdata in enumerate(validation_loader):
                    vinputs, vtargets = vdata
                    voutputs = model(vinputs)
                    vloss = float(torch.mean(torch.abs(voutputs[-1] - vtargets)))
                    running_vloss += vloss
          
          avg_vloss = running_vloss / (i + 1)
          print('LOSS train {} a.u., valid {} a.u.'.format(avg_loss, avg_vloss))
          stop=time()
          print(stop-start,"s")
          # Log the running loss averaged per batch for both training and validation
          f = open("training_loss_xPaiNN_v4_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(formula,increments,basis,n_interactions,n_atom_basis,cut,radial_basis,cutoff_fn,n_SchNOrb), "a")
          g = open("validation_loss_xPaiNN_v4_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(formula,increments,basis,n_interactions,n_atom_basis,cut,radial_basis,cutoff_fn,n_SchNOrb), "a")
          f.write("{},".format(avg_loss))
          g.write("{},".format(avg_vloss))
          f.close()
          g.close()

          # save the best model's state
          if avg_loss<best_loss and save:
               print("save")
               best_loss = avg_loss
               model_path = 'xPaiNN_v4_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(formula,increments,basis,n_interactions,n_atom_basis,cut,radial_basis,cutoff_fn,n_SchNOrb)
               torch.save(model.state_dict(), model_path)

     return avg_loss

def test_diatomics(model,increments):
    plot_lo = 0.7
    plot_hi = 2
    plot_increments = 200
    n_atoms = 2

    positions = np.zeros((plot_increments,n_atoms,3))
    h_s = np.zeros((plot_increments,B,B))
    energies = np.zeros(plot_increments)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # 
        print("R =",R)
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
    AE = (E.detach().numpy()-energies)*27.2114/0.0434
    MAE = np.mean(np.abs(AE))
    print("MAE",MAE,"kcal/mol")

    plt.plot(np.linspace(0.7,2,increments)[training_indices],[0]*90,"o",color="k",label="")
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),AE)
    plt.xlabel("atomic distance in Angstrom")
    plt.ylabel(r"absolute error $|E_{\text{pred}}-E_{\text{ref}}|$ in kcal/mol")
    plt.legend()
    plt.show()
    return

def SchNOrb_data(formula,indices,increments,label):
    if formula=='C2OH6':
        path_to_db = "./schnorb_hamiltonian_ethanol_hf.db"
    elif formula=='C3O2H4':
        path_to_db = "./schnorb_hamiltonian_malondialdehyde.db"
    database = AtomsData(path_to_db)
    numbers = database.get_properties(label)[1]["_atomic_numbers"]
    positions0 = database.get_properties(label)[1]["_positions"]
    B, B_k, N, M = get_basis_size(basis,numbers)
    F_offset, E_HF = RHF_dissociation(formula,basis,numbers,positions0)
    E_ref = database.get_properties(label)[1]["energy"][0]
    delta_E = torch.tensor(E_ref - E_HF)

    positions = torch.stack([torch.tensor(database.get_properties(int(str(i)))[1]["_positions"]) for i in indices]).to(dtype=torch.double)
    energies = torch.stack([torch.tensor(database.get_properties(int(str(i)))[1]["energy"][0]) for i in indices]).to(dtype=torch.double)
    h_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    F_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    S_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    return numbers, positions, F_matrices, S_matrices, h_matrices, energies, F_offset, delta_E


def diatomics_data(formula,basis,increments):
    if formula=='H2':
        numbers = [1,1]
        lo = 0.4
        hi = 1.3
        R_infty = 2
    if formula=='N2':
        numbers = [7,7]
        lo = 0.7
        hi = 2
        R_infty = 5
    if formula=='CO':
        numbers = [8,6]
        lo = 0.7
        hi = 2
        R_infty = 2
    B, B_k, N, M = get_basis_size(basis,numbers)
    F_offset, _ = RHF_dissociation(formula,basis,numbers,np.array([[0,0,0],[R_infty,0,0]]))
    _,U = np.linalg.eigh(F_offset)
    
    positions = torch.zeros((increments,M,3))
    energies = torch.zeros(increments)
    h_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    F_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    S_matrices = torch.zeros((increments,B,B),dtype=torch.double)
    
    data = np.linspace(lo,hi,increments) # np.random.rand(increments)*(hi-lo) + lo # 
    for i,R in enumerate(data):
        positions_i = [[0,0,0],[R,0,0]]
        positions_i = list(map(np.array,positions_i))
        
        mol = gto.M(atom = [[numbers[l],positions_i[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()

        h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_i = mol.intor('int1e_ovlp')
        F_i = mf.get_fock()
        positions[i] = torch.tensor(positions_i)
        energies[i] = mf.e_tot
        F_matrices[i,:,:] = torch.tensor(F_i)
        h_matrices[i,:,:] = torch.tensor(h_i)
        S_matrices[i,:,:] = torch.tensor(S_i)
        
    return numbers, positions, F_matrices, S_matrices, h_matrices, energies, F_offset, torch.tensor(0)


if __name__=="__main__"():
    # learning parameters
    np.random.seed(1)
    formula='N2' # 'C3O2H4' # 'C2OH6' # 
    basis = '631g' # 'def2-svp' # 
    n_interactions=1
    n_atom_basis=8
    cut=5
    radial_basis='Gaussian'
    cutoff_fn='cosine'
    n_SchNOrb=8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    EPOCHS = 3000
    fixed = False
    increments = 100
    label = 0


    indices = np.arange(increments)
    training_indices = np.random.choice(indices,int(0.9*increments),replace=False) # np.concatenate((np.arange(increments)[:int((0.9-lo)/(hi-lo)*increments)],np.arange(increments)[int((1.3-lo)/(hi-lo)*increments)+1:])) # np.array([0,3,6,9,12,21,30,39,48,59]) # np.random.choice(np.arange(increments),int(0.9*increments),replace=False) # 
    validation_indices = np.setdiff1d(indices,training_indices) # np.random.choice(np.setdiff1d(np.arange(increments),training_indices),int(0.1*increments),replace=False)
    #numbers, positions, F_matrices, S_matrices, h_matrices, energies, F_offset, delta_E = SchNOrb_data(formula,indices,increments,0)
    numbers, positions, F_matrices, S_matrices, h_matrices, energies, F_offset, delta_E = diatomics_data(formula,basis,increments)

    B, B_k, N, M = get_basis_size(basis,numbers)
    converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=10))
    input = converter(Atoms(numbers=numbers, positions=positions[label]))
    n_atoms = len(numbers)
    idx_i = input["_idx_i"]
    idx_j = input["_idx_j"]
    neighbors = idx_j.reshape((M,M-1))
    if fixed:
        h0_new = compute_h_matrix(basis,numbers,positions[label,None,...],device).reshape((B,B))
        nuc0 = torch.tensor(sum([numbers[k]*numbers[l]/np.linalg.norm(positions[label,k]-positions[label,l])*a0 for k in range(n_atoms) for l in range(k)]),
                            dtype=torch.float,
                            device=device)

    # create data loaders
    print("build training data...")
    training_data   = SchNet_XAIDataSet(positions,energies,training_indices)
    print("build test data...")
    validation_data = SchNet_XAIDataSet(positions,energies,validation_indices)
    training_loader = torch.utils.data.DataLoader(training_data,batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size)

    # create explanation model, loss and optimizer
    model = xPaiNN(basis,F_offset,
                    n_interactions=n_interactions,
                    n_atom_basis=n_atom_basis,
                    cut=cut,
                    radial_basis=radial_basis,
                    cutoff_fn=cutoff_fn,
                    n_SchNOrb=n_SchNOrb,
                    fixed=fixed,delta_E=delta_E)
    model.h_available()
    model.to(torch.double)
    model.load_state_dict(torch.load('xPaiNN_v4_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(formula,increments,basis,n_interactions,n_atom_basis,cut,radial_basis,cutoff_fn,n_SchNOrb),map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load('xPaiNN_v4_2602855_mda_def2-svp_fixed',map_location=torch.device('cpu'))) # xPaiNN_v4_C3O2H4_26978_def2-svp_0_3_128_5_Gaussian_cosine_128_fixed
    loss_fn = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)

    print("number of parameters:",number_of_parameters(model))
    #train(model)
    test_diatomics(model,increments)
