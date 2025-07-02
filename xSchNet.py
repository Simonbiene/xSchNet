import numpy as np

import scipy
from scipy.stats import t, f

import torch
from torch import nn

import schnetpack as spk

from pyscf import scf, gto

from ase import Atoms
from utils.utils import *
import matplotlib.pyplot as plt


a0 = 0.529177210544 # in Angstrom
Hartree_to_eV = 27.211386256
Hartree_to_kcal_per_mol = 0.04338
np.random.seed(1)

class SchNOrbInteraction(nn.Module):

    def __init__(self,n_factors,n_rbf,idx_j,neighbors,n_atoms):
        super(SchNOrbInteraction, self).__init__()

        self.idx_j = idx_j
        self.neighbors = neighbors
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
        xj = torch.gather(x,1,self.idx_j[None,:,None].expand(batch_size,-1,self.n_factors)).reshape((batch_size,self.n_atoms,self.n_atoms-1,self.n_factors))
        h_ij = torch.einsum('nki,nkli->nkli', x, torch.einsum('nkli,nkli->nkli',W,xj) )

        # get neighbors
        nbh = self.neighbors[None,:,:,None].expand((batch_size,self.n_atoms,self.n_atoms-1,self.n_factors))

        p_ij_pair = self.pairnet(h_ij)
        p_ij_env = self.envnet(h_ij)
        p_i_env = torch.sum(p_ij_env,dim=2,keepdim=True)
        p_j_env = torch.gather(p_i_env.expand(-1,-1,self.n_atoms-1,-1),1,nbh)
        p_ij = p_ij_pair + p_j_env + p_i_env
    
        v = self.f2out(torch.sum(h_ij,dim=2))

        return p_ij, v

class NoLearning_model(nn.Module):
    def __init__(self,F_dissoc):
        super(NoLearning_model, self).__init__()
        self.F_dissoc = F_dissoc

    def forward(self,input):
        #positions = list(map(lambda R: np.array([[0,0,0],[R,0,0]]),input[0]))
        h_new = input[2]

        e, U = torch.linalg.eigh(self.F_dissoc)
        
        # compute electronic energy
        E = torch.sum(e[:N//2]) + torch.einsum('nii->n',torch.einsum('pi,npj->nij',U[:,:N//2],torch.einsum('npq,qj->npj', h_new, U[:,:N//2])))
        return self.F_dissoc, e, E 


class xSchNet(nn.Module):
    '''
    Creates a model that concatenates SchNet with an interpretable prediction head. SchNet is now part of the model and is trained with it.
    '''
    def __init__(self,
                 basis,F_dissoc,
                 n_interactions=1,
                 n_SchNet=5,
                 cut=10,
                 radial_basis='Gaussian',
                 cutoff_fn_name='cosine',
                 n_SchNOrb=5):
        super(xSchNet, self).__init__()
        
        self.n_interactions = n_interactions
        self.n_SchNet = n_SchNet
        self.cut = cut
        self.radial_basis = radial_basis
        self.cutoff_fn_name = cutoff_fn_name
        self.n_SchNOrb = n_SchNOrb

        B, B_k, N, M = get_basis_size(basis,numbers)

        self.n_atoms = M
        self.B = B
        self.B_k = B_k
        self.Bmax = max([B_k[Z_k-1] for Z_k in numbers])
        if formula=='H2O' and basis=="sto-6g":
            self.mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1],
                                    [1, 1, 1, 0, 0, 1, 1],
                                    [1, 1, 1, 0, 0, 1, 1],
                                    [0, 0, 0, 1, 0, 1,-1],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [1, 1, 1, 1, 0, 1, 1],
                                    [1, 1, 1,-1, 0, 1, 1]]) # STO-6G for H2O
        else:
            self.mask = get_mask(basis,B)
        self.register_buffer('F_dissoc', F_dissoc)

        if radial_basis=='Gaussian':
            self.rbf = spk.nn.GaussianRBF(n_rbf=20, cutoff=cut)
        elif radial_basis=='Bessel':
            self.rbf = spk.nn.BesselRBF(n_rbf=20, cutoff=cut)

        if cutoff_fn_name=='cosine':
            self.cutoff_fn = spk.nn.CosineCutoff(cutoff=cut)
        elif cutoff_fn_name=='mollifier':
            self.cutoff_fn = spk.nn.MollifierCutoff(cutoff=cut)

        self.embedding = nn.Embedding(10, n_SchNet)
        
        self.interactions = spk.nn.replicate_module(
            lambda: spk.representation.SchNetInteraction(
                n_atom_basis=n_SchNet,
                n_rbf=self.rbf.n_rbf,
                n_filters=n_SchNet,
            ),
            n_interactions,
            share_params=False,
        )

        self.transfer = nn.Linear(n_SchNet,n_SchNOrb)

        # self.WF_on = nn.Linear(n_SchNOrb,self.Bmax**2)
        # self.WF_off = nn.Linear(n_SchNOrb,self.Bmax**2)
        self.WF_on = nn.Sequential(nn.Linear(n_SchNOrb,n_SchNOrb),
                                   ShiftedSoftplus(),
                                   nn.Linear(n_SchNOrb,self.Bmax**2))
        self.WF_off = nn.Sequential(nn.Linear(n_SchNOrb,n_SchNOrb),
                                   ShiftedSoftplus(),
                                   nn.Linear(n_SchNOrb,self.Bmax**2))

        return
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self,input):
        positions = input[0]
        h_new = input[1]
        batch_size = h_new.shape[0]
        
        '''
        SchNet interactions
        '''
        # pre-processing: compute pair-features
        Z = torch.stack([atomic_numbers for i in range(batch_size)])
        R = positions.type(torch.float)
        r_ij = torch.stack([R[n,idx_i]-R[n,idx_j] for n in range(batch_size)])
        d_ij = torch.norm(r_ij, dim=2)
        f_ij = self.rbf(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)
        
        # compute initial embeddings
        x = self.embedding(Z)
        
        # compute interaction blocks and update atomic embeddings
        for interaction in self.interactions:
            v = torch.stack([interaction(x[n], f_ij[n], idx_i, idx_j, rcut_ij[n]) for n in range(batch_size)])
            x = x + v
            
        x = self.transfer(x)
        
        '''
        SchNOrb interactions
        '''
        
        # do pairwise interactions
        p_ij = torch.einsum('nki,nli->nkli',x,x)

        # construct the electron repulsion matrix
        J_K = torch.zeros((batch_size,self.B,self.B))
        n_k = 0
        for k, Z in enumerate(numbers):
            Bk = self.B_k[Z-1]

            # get on-diagonal block-matrices
            F_on = torch.sum(self.WF_on(p_ij[:,k,:,:]),dim=1).reshape((batch_size,self.Bmax,self.Bmax))
            
            # symmetrize
            F_on = 0.5*(F_on + F_on.transpose(1,2))
            
            J_K[:,n_k:n_k+Bk,n_k:n_k+Bk] = F_on[:,:Bk,:Bk]
            n_l = 0
            for l in range(k):
                Bl = self.B_k[numbers[l]-1]

                # get off-diagonal matrix
                F_off = self.WF_off(p_ij[:,k,l,:]).reshape((batch_size,self.Bmax,self.Bmax))
                
                # symmetrize, if atoms are indistinguishable (=have same distances to neighbors and have same type)
                F_off = 0.5*(F_off + F_off.transpose(1,2)) + F_off*torch.mean((x[:,k,:]-x[:,l,:]),dim=1).reshape((batch_size,1,1))
                
                # fill the overall matrix
                J_K[:,n_k:n_k+Bk,n_l:n_l+Bl] = F_off[:,:Bk,:Bl]
                J_K[:,n_l:n_l+Bl,n_k:n_k+Bk] = F_off[:,:Bk,:Bl].transpose(1,2)
                    
                n_l += Bl

            n_k += Bk
        
        # apply mask and offset
        zeros = torch.zeros(h_new.shape)
        zeros[torch.abs(h_new)>1e-7] = 1 # Warning: can create discontinuities, if a non-zero value of h_new drops below the threshold
        J_K = zeros * self.mask * J_K
        F_new = J_K + self.F_dissoc
        if F_new.requires_grad:
            F_new.retain_grad()
        
        # solve Fock equation
        e, U = torch.linalg.eigh(F_new)
        
        # compute electronic energy
        E_e = torch.sum(e[:,:N//2],dim=1) + torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', h_new, U[:,:,:N//2])))
        nuc = torch.sum(torch.stack([numbers[k]*numbers[l]/torch.norm(positions[:,k]-positions[:,l],dim=1)*a0 
                                     for k in range(self.n_atoms) for l in range(k)],dim=-1),dim=-1)
        return F_new, e, E_e + nuc
    

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
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, n):
        input = (self.input_data[n], self.h_new[n,:,:])
        target = self.target_E[n] # ,self.h_new[n,:,:],self.J_K_new[n,:,:])
        return input, target


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self,output,target):
         '''
         Loss terms for:
         -energy
         -occupied states (sould be negative)
         -HOMO-LUMO gap (should be >0.1 to avoid corssings)
         '''
         F_new, e, E = output
         return torch.mean( (target - E)**2 ) + 100*torch.sum(torch.relu(0.1 - e[:,N//2] + e[:,N//2-1])) # + torch.sum(torch.relu(e[:,:N//2])) 


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
          running_loss += energy_loss # loss.item() # 

     return running_loss / (i + 1)


def train(model,save_number=None,save_name=None,save=True):
     best_loss = 1_000_000.

     for epoch in range(EPOCHS):
          print('EPOCH {}:'.format(epoch + 1))
          
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

          # Log the running loss averaged per batch for both training and validation
          f = open("training_loss_xSchNet_v2_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(formula,increments,basis,model.n_interactions,model.n_SchNet,model.cut,model.radial_basis,model.cutoff_fn_name,model.n_SchNOrb), "a")
          g = open("validation_loss_xSchNet_v2_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(formula,increments,basis,model.n_interactions,model.n_SchNet,model.cut,model.radial_basis,model.cutoff_fn_name,model.n_SchNOrb), "a")
          f.write("{},".format(avg_loss))
          g.write("{},".format(avg_vloss))
          f.close()
          g.close()

          # save the best model's state
          if avg_loss<best_loss and save:
               print("save")
               best_loss = avg_loss
               if save_number==None:
                   model_path = 'xSchNet_v2_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(formula,increments,basis,model.n_interactions,model.n_SchNet,model.cut,model.radial_basis,model.cutoff_fn_name,model.n_SchNOrb)
               else:
                   model_path = 'xSchNet_v2_{}_{}_{}_{}_trial_{}'.format(formula,increments,basis,save_name,save_number)
               torch.save(model.state_dict(), model_path)


     return avg_loss

def extrapolation_plots():
    """
    Code used to generate Figure 4.7 a).
    """
    plot_lo = 0.7
    plot_hi = 3
    plot_increments = 200
    sample_sizes = [10,20,50,100,200,500]
    #trials = [33,0,41,49,27,9] # N2
    #trials = [4,22,0,21] # CO
    colors = ['C0','C1','C3','C4','C5','C6','C7']

    # compute reference values and matrices
    E_values = np.zeros(plot_increments)
    positions_all = torch.zeros(plot_increments,M,3,dtype=torch.double)
    h_new = torch.zeros(plot_increments,B,B,dtype=torch.double)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)):
        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()

        E_values[i] = mf.e_tot

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
        S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)).real)
        h_new[i,:,:] = torch.einsum('pr,rq->pq',S_inv_root,torch.einsum('rs,sq->rq', h, S_inv_root))
        positions_all[i,:,:] = torch.tensor(positions,dtype=torch.double)

    E_min = E_values # np.min(E_values)
    R_values = np.linspace(plot_lo,plot_hi,plot_increments)[E_values!=0]
    plt.plot(R_values,(E_values-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,lw=2,color='k',label="HF calculation")

    # load model and compute prediction values
    for i, n in enumerate(sample_sizes):
        model = xSchNet(basis,F_dissoc)
        model.load_state_dict(torch.load('SchNOrb_v8_2_{}_{}_{}_success'.format(formula,2*n,basis)))

        F_new, e, E = model((positions_all,S_inv_root,h_new))
        predictions = E.detach().numpy()
        plt.plot(R_values,(predictions-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,lw=2,color=colors[i].format(i),label="{} training points".format(n))
    
    # linear model benchmark
    model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear_success'.format(formula,basis)))

    F_new, e, E = model((positions_all,S_inv_root,h_new))
    predictions = E.detach().numpy()
    plt.plot(R_values,(predictions-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,lw=2,color=colors[i+1],label="linear model")
    #plt.plot(data[training_indices],(target_E_test[training_indices]-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,"o",color='k',label='training points')
    plt.plot([2,2],[-320,320],"--",color='k')
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel("Energy difference in kcal/mol",fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.legend(fontsize=22)
    plt.show()
    return

def extrapolation_density_plots():
    """
    Code used to generate Figure 4.7 b).
    """
    plot_lo = 0.7
    plot_hi = 3
    plot_increments = 200
    sample_sizes = [10,20,50,100,200,500]
    #trials = [33,0,41,49,27,9]
    colors = ['C0','C1','C3','C4','C5','C6','C7']

    E_values = np.zeros(plot_increments)
    v = torch.zeros((plot_increments,B,B,B,B),dtype=torch.double)
    positions_all = torch.zeros(plot_increments,M,3,dtype=torch.double)
    S_inv_root = torch.zeros(plot_increments,B,B,dtype=torch.double)
    h_new = torch.zeros(plot_increments,B,B,dtype=torch.double)
    S4 = np.zeros((plot_increments,B,B,B,B))
    D_ref = np.zeros((plot_increments,B,B))
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)):
        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        counter = 0
        while mf.e_tot > E_dissoc and R>2 and counter<5:
            # if often not converged, then skip and don't save the energy
            counter+=1
            mf = scf.RHF(mol)
            mf.kernel()
        if counter==5:
            continue
        E_values[i] = mf.e_tot - mf.energy_nuc()

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
        S_inv_root_i = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)).real)
        v_new = np.einsum('ap,pbcd->abcd',S_inv_root_i,np.einsum('bq,pqcd->pbcd',S_inv_root_i,np.einsum('cr,pqrd->pqcd',S_inv_root_i,np.einsum('ds,pqrs->pqrd', S_inv_root_i, mol.intor('int2e')))))
        v[i,:,:,:,:] = torch.tensor(v_new)
        h_new[i,:,:] = torch.einsum('pr,rq->pq',S_inv_root_i,torch.einsum('rs,sq->rq', h, S_inv_root_i))
        S_inv_root[i,:,:] = S_inv_root_i
        positions_all[i,:,:] = torch.tensor(positions,dtype=torch.double)
        if formula=='N2':
            S4[i,:,:,:,:] = torch.load("{}_four_center_ovlp_{}_R_{}".format(formula,basis,round(R*1000))).detach().numpy()
        elif formula=='CO':
            S4[i,:,:,:,:] = torch.load("temp/{}_four_center_ovlp_{}_R_{}".format(formula,basis,round(R*1000))).detach().numpy()
        D_ref[i,:,:] = mf.mo_coeff[:,:N//2] @ mf.mo_coeff[:,:N//2].transpose()

    R_values = np.linspace(plot_lo,plot_hi,plot_increments)[E_values!=0]
    for i, n in enumerate(sample_sizes):
        model = xSchNet(basis,F_dissoc)
        model.load_state_dict(torch.load('SchNOrb_v8_2_{}_{}_{}_success'.format(formula,2*n,basis)))

        F_new, e, E = model((positions_all,S_inv_root,h_new))
        _, U = torch.linalg.eigh(F_new)
        P = (S_inv_root @ U[:,:,:N//2]).detach().numpy()
        D = np.einsum('npi,nqi->npq',P,P)
        
        # J = torch.einsum('npqjj->npq',torch.einsum('nri,npqrj->npqij',U[:,:,:N//2],torch.einsum('npqrs,nsj->npqrj', v, U[:,:,:N//2])))
        # K = torch.einsum('npjjs->nps',torch.einsum('nqi,npqjs->npijs',U[:,:,:N//2],torch.einsum('npqrs,nrj->npqjs', v, U[:,:,:N//2])))
        # E_ee = torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', 2*h_new + 2*J - K, U[:,:,:N//2])))
        # error = (E_ee.detach().numpy() - E_values)*Hartree_to_eV/Hartree_to_kcal_per_mol
        # error = (E.detach().numpy() - E_values)*Hartree_to_eV/Hartree_to_kcal_per_mol
        error = np.sqrt(np.einsum('npq,npq->n',D,np.einsum('npqrs,nrs->npq',S4,D)) + np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D_ref)) -\
                        2*np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D)))
        plt.plot(R_values,error,lw=2,color=colors[i].format(i),label="{} training points".format(n))
    
    model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear_success'.format(formula,basis)))

    F_new, e, E = model((positions_all,S_inv_root,h_new))
    _, U = torch.linalg.eigh(F_new)
    P = (S_inv_root @ U[:,:,:N//2]).detach().numpy()
    D = np.einsum('npi,nqi->npq',P,P)
    error = np.sqrt(np.einsum('npq,npq->n',D,np.einsum('npqrs,nrs->npq',S4,D)) + np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D_ref)) -\
                    2*np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D)))
    plt.plot(R_values,error,lw=2,color=colors[i+1],label="linear model")
    #plt.plot(data[training_indices],(target_E_test[training_indices]-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,"o",color='k',label='training points')
    plt.plot([2,2],[0,0.31],"--",color='k')
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel(r"error $d_{L^2}$ in $1/a_0^{-3/2}$",fontsize=22)
    plt.ylim([0,0.32])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.legend(fontsize=22)
    plt.show()
    return


def interpolation_plots():
    plot_lo = 0.7
    plot_hi = 2
    plot_increments = 200
    #trials = [33,0,41,49,27,9] # N2
    #trials = [4,22,0,21] # CO
    colors = ['C0','C1','C3','C4','C5','C6']

    E_values = np.zeros(plot_increments)
    positions_all = torch.zeros(plot_increments,M,3,dtype=torch.double)
    h_new = torch.zeros(plot_increments,B,B,dtype=torch.double)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)):
        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        
        E_values[i] = mf.e_tot

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
        S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)).real)
        h_new[i,:,:] = torch.einsum('pr,rq->pq',S_inv_root,torch.einsum('rs,sq->rq', h, S_inv_root))
        positions_all[i,:,:] = torch.tensor(positions,dtype=torch.double)

    E_min = E_values # np.min(E_values)
    R_values = np.linspace(plot_lo,plot_hi,plot_increments)[E_values!=0]

    # normal model
    model = xSchNet(basis,F_dissoc)
    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_21'.format(formula,basis)))
    F_new, e, E = model((positions_all,S_inv_root,h_new))
    predictions = E.detach().numpy()
    plt.plot(R_values,(predictions-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,lw=2,label="normal model")
    print("MAE training:",np.mean(np.abs(predictions-E_min)[training_indices])*Hartree_to_eV/Hartree_to_kcal_per_mol)
    print("MAE test:",np.mean(np.abs(predictions-E_min)[validation_indices])*Hartree_to_eV/Hartree_to_kcal_per_mol)
    
    # interpolation model
    model = xSchNet(basis,F_dissoc)
    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}'.format(formula,basis)))
    F_new, e, E = model((positions_all,S_inv_root,h_new))
    predictions = E.detach().numpy()

    print("MAE training:",np.mean(np.abs(predictions-E_min)[training_indices])*Hartree_to_eV/Hartree_to_kcal_per_mol)
    print("MAE test:",np.mean(np.abs(predictions-E_min)[validation_indices])*Hartree_to_eV/Hartree_to_kcal_per_mol)
    plt.plot(R_values,(predictions-E_min)*Hartree_to_eV/Hartree_to_kcal_per_mol,lw=2,label="interpolation model")
    plt.plot(data[training_indices],np.zeros(increments)[training_indices],"o",color='k',label='training points')
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel("Energy in kcal/mol",fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()
    return


def compare_bond_orders():
    plot_lo = 0.7
    plot_hi = 2
    plot_increments = 200
    sample_sizes = [10,20,100,500]
    trials = [33,0,49,9] # [33,0,41,49,27,9]
    colors = ['C0','C1','C3','C4','C5','C6']

    BO_Löwdin_ref = np.linspace(plot_lo,plot_hi,plot_increments)
    positions_all = torch.zeros(plot_increments,M,3,dtype=torch.double)
    h_new = torch.zeros(plot_increments,B,B,dtype=torch.double)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)):
        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc')).reshape((1,B,B))
        S_root = scipy.linalg.sqrtm(S)
        S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)).real).reshape((1,B,B))
        h_new[i,:,:] = torch.einsum('npr,nrq->npq',S_inv_root,torch.einsum('nrs,nsq->nrq', h, S_inv_root))
        positions_all[i,:,:] = torch.tensor(positions,dtype=torch.double)

        D = 2 * (S_root@mf.mo_coeff[:,:N//2]) @ (mf.mo_coeff[:,:N//2].transpose()@S_root)
        print("Löwdin bond index reference:",np.sum(D[:B//2,B//2:]**2))
        BO_Löwdin_ref[i] = np.sum(D[:B//2,B//2:]**2)
        
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),BO_Löwdin_ref,color='k',label="HF calculation")
    for i, n in enumerate(sample_sizes):
        model = xSchNet(basis,F_dissoc)
        model.load_state_dict(torch.load('SchNOrb_v8_2_{}_{}_{}__trial_{}'.format(formula,2*n,basis,trials[i])))

        F_new, e, E = model((positions_all,S_inv_root,h_new))
        _, U = torch.linalg.eigh(F_new)
        #P = (S_inv_root.reshape((B,B)) @ L).detach().numpy()
        #F_new = F_new.detach().numpy().reshape((B,B))
        #F = np.einsum('pr,rq->pq',S_root,np.einsum('rs,sq->rq', F_new, S_root))
        #_, U = scipy.linalg.eigh(F,S)

        # Löwdin population
        D = 2* U[:,:,:N//2]@U[:,:,:N//2].transpose(1,2)
        BO_Löwdin_pred = torch.sum(D[:,:B//2,B//2:]**2,dim=(1,2)).detach().numpy()
        plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),BO_Löwdin_pred,color=colors[i],lw=2,label="{} training points".format(n))
    
    model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear_success'.format(formula,basis)))
    F_new, e, E = model((positions_all,S_inv_root,h_new))
    _, U = torch.linalg.eigh(F_new)
    D = 2* U[:,:,:N//2]@U[:,:,:N//2].transpose(1,2)
    BO_Löwdin_pred = torch.sum(D[:,:B//2,B//2:]**2,dim=(1,2)).detach().numpy()
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),BO_Löwdin_pred,color="C8",lw=2,label="linear model")
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel("Bond order",fontsize=22)
    plt.ylim([1,4])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()
    return
    

def density_comparison(experiment):
    """
    Code used to generate left-hand-sides of Figure 4.10, 4.12, 4.18, 4.20, i.e., the average physical error according to equation (4.5). 
    The input experiment takes values 'sample size' and 'hyperparameters' depending on whether models trained on different sample sizes 
    or models with different hyperparameters should be compared.
    """
    plot_lo = lo
    plot_hi = hi
    plot_increments = 200
    colors = ['C0','C1','C3','C4','C5','C6']

    E_values = torch.zeros(plot_increments)
    positions_list = torch.zeros(plot_increments,M,3,dtype=torch.double)
    S_root = torch.zeros(plot_increments,B,B,dtype=torch.double)
    S_inv_root = torch.zeros(plot_increments,B,B,dtype=torch.double)
    h = torch.zeros(plot_increments,B,B,dtype=torch.double)
    v = torch.zeros(plot_increments,B,B,B,B,dtype=torch.double)
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # enumerate(np.arange(1,3.2,0.25)): # 
        print("R =",R)

        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))
        positions_list[i,:,:] = torch.tensor(positions)

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        E_values[i] = mf.e_tot - mf.energy_nuc()

        S = mol.intor('int1e_ovlp')
        h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_inv_root_i = np.linalg.inv(scipy.linalg.sqrtm(S))
        h_new = np.einsum('pr,rq->pq',S_inv_root_i,np.einsum('rs,sq->rq', h_i, S_inv_root_i))
        v_new = np.einsum('ap,pbcd->abcd',S_inv_root_i,np.einsum('bq,pqcd->pbcd',S_inv_root_i,np.einsum('cr,pqrd->pqcd',S_inv_root_i,np.einsum('ds,pqrs->pqrd', S_inv_root_i, mol.intor('int2e')))))
        S_root[i,:,:] = torch.tensor(scipy.linalg.sqrtm(S))
        S_inv_root[i,:,:] = torch.tensor(S_inv_root_i)
        h[i,:,:] = torch.tensor(h_new)
        v[i,:,:,:,:] = torch.tensor(v_new)
        

    # no-learning-model
    _, U = torch.linalg.eigh(F_dissoc)
    J = torch.einsum('npqjj->npq',torch.einsum('ri,npqrj->npqij',U[:,:N//2],torch.einsum('npqrs,sj->npqrj', v, U[:,:N//2])))
    K = torch.einsum('npjjs->nps',torch.einsum('qi,npqjs->npijs',U[:,:N//2],torch.einsum('npqrs,rj->npqjs', v, U[:,:N//2])))
    E_ee = torch.einsum('nii->n',torch.einsum('pi,npj->nij',U[:,:N//2],torch.einsum('npq,qj->npj', 2*h + 2*J - K, U[:,:N//2])))
    error = E_ee - E_values
    error = Hartree_to_eV/Hartree_to_kcal_per_mol*error.detach().numpy()
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),error,label="no learning",lw=2,color='k')


    trials = 100
    sample_size = np.zeros(20,dtype=int)
    avg_error = np.zeros((20,plot_increments))
    var_error = np.zeros((20,plot_increments))
    if experiment=='sample size':
        if formula=='CO':
            tests = range(5)
        else:
            tests = range(6)
    else:
        tests = [11,12,13,14,15,16]
    for c, j in enumerate(tests): # enumerate(range(6)): # enumerate([0,1,3,5,18]): # enumerate([11,12,13,14,15,16]): # 
        print("trial",j)
        try:
            error = torch.zeros((plot_increments,trials))
            for i in range(trials):
                if j==0:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_20_{}__trial_{}'.format(formula,basis,i)))
                    label='10 data points'
                elif j==1:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}__trial_{}'.format(formula,basis,i)))
                    label='20 data points'
                elif j==2:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,i)))
                    label='50 data points'
                elif j==3:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i)))
                    label='100 data points'
                elif j==4:
                    if formula=='CO':
                        explanation_model = xSchNet(basis,F_dissoc,gpu=True)
                        explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                        label='200 data points'
                    else:
                        explanation_model = xSchNet(basis,F_dissoc)
                        explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                        label='200 data points'
                elif j==5:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_1000_{}__trial_{}'.format(formula,basis,i)))
                    label='500 data points'
                elif j==6:
                    explanation_model = xSchNet(basis,F_dissoc,n_SchNet=32,n_SchNOrb=32)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_features_trial_{}'.format(formula,basis,i)))
                    label='smaller feature dim.'
                elif j==7:
                    explanation_model = xSchNet(basis,F_dissoc,lmax=1)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_lmax_trial_{}'.format(formula,basis,i)))
                    label='less SchNOrb layers'
                elif j==8:
                    explanation_model = xSchNet(basis,F_dissoc,lmax=3)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_lmax_trial_{}'.format(formula,basis,i)))
                    label='more SchNOrb layers'
                elif j==9:
                    explanation_model = xSchNet(basis,F_dissoc,n_interactions=1)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_less_SchNet_interact_trial_{}'.format(formula,basis,i)))
                    label='less SchNet layers'
                elif j==10:
                    explanation_model = xSchNet(basis,F_dissoc,n_interactions=3)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_interact_trial_{}'.format(formula,basis,i)))
                    label='more SchNet layers'
                elif j==11:
                    try:
                        # trained on laptop
                        explanation_model = xSchNet(basis,F_dissoc)
                        explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i)))
                    except Exception:
                        # trained on hydra
                        explanation_model = xSchNet(basis,F_dissoc,gpu=True)
                        explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='baseline'
                elif j==12:
                    explanation_model = xSchNet(basis,F_dissoc,n_interactions=2,gpu=True)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='2 SchNet layers'
                elif j==13:
                    explanation_model = xSchNet(basis,F_dissoc,lmax=2,gpu=True)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNOrb_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label=r'$l_{\text{max}}=2$'
                elif j==14:
                    explanation_model = xSchNet(basis,F_dissoc,n_SchNet=3,n_SchNOrb=3,gpu=True)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature dim. 3'
                elif j==15:
                    explanation_model = xSchNet(basis,F_dissoc,n_SchNet=8,n_SchNOrb=8,gpu=True)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature dim. 8'
                elif j==16:
                    explanation_model = xSchNet(basis,F_dissoc,n_SchNet=64,n_SchNOrb=64,gpu=True)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_feature_dim_64_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature dim. 64'
                elif j==17:
                    explanation_model = xSchNet(basis,F_dissoc,n_SchNet=64,n_SchNOrb=64,n_interactions=2,lmax=2,old=True)
                    explanation_model.load_state_dict(torch.load('large_xSchNet/SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='large xSchNet'
                elif j==18:
                    explanation_model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear__trial_{}'.format(formula,basis,i)))
                    label='linear model'
                elif j==19:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__offset_eq_trial_{}'.format(formula,basis,i)))
                    label='offset matrix at R=1.1 Angstrom'
                
                running_loss = 0
                with torch.no_grad():
                    # for m, data in enumerate(training_loader):
                    #     inputs, vtargets = data
                    #     outputs = explanation_model(inputs)
                    #     loss = float(torch.mean(torch.abs(outputs[-1] - vtargets)))
                    #     running_loss += loss
                
                    # avg_loss = running_loss / (m + 1)
                    # if avg_loss>Hartree_to_kcal_per_mol/Hartree_to_eV:
                    #     continue

                    F_new, e, E = explanation_model((positions_list,None,h))
                    _, U = torch.linalg.eigh(F_new)
                    J = torch.einsum('npqjj->npq',torch.einsum('nri,npqrj->npqij',U[:,:,:N//2],torch.einsum('npqrs,nsj->npqrj', v, U[:,:,:N//2])))
                    K = torch.einsum('npjjs->nps',torch.einsum('nqi,npqjs->npijs',U[:,:,:N//2],torch.einsum('npqrs,nrj->npqjs', v, U[:,:,:N//2])))
                    E_ee = torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', 2*h + 2*J - K, U[:,:,:N//2])))
                    error[:,i] = E_ee - E_values
        except Exception:
            pass
        
        error = Hartree_to_eV/Hartree_to_kcal_per_mol*error
        sample_size[j] = int(torch.sum(error[0,:]>0))
        print(sample_size[j])
        avg_error[j,:] = torch.mean(error[:,error[0,:]>0],dim=1).detach().numpy()
        var_error[j,:] = torch.var(error[:,error[0,:]>0],dim=1).detach().numpy()

        # get average orbitals by diagonalizing the average density matrix
        # D = S_root @ sum_D/(trials-1) @ S_root
        # d, U = torch.linalg.eigh(D)
        # P = S_inv_root @ U
        # mol = gto.M(atom = [[numbers[l],positions_list[-1,l,:]] for l in range(len(numbers))],
        #             basis = basis,
        #             unit='Angstrom')
        # mlo = PM(mol)
        # mo0 = P[-1,:,-N//2:].detach().numpy()
        # P = mlo.kernel(mo0)
        # print(d[-1])
        # print(P)
        
        plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),avg_error[j,:],label=label,lw=2,color=colors[c%6])
        # if j==18:
        #     q = t.ppf(0.975,sample_size[j]-1)
        #     plt.fill_between(np.linspace(plot_lo,plot_hi,plot_increments),
        #                     avg_error[j,:] - q*np.sqrt(var_error[j,:])/np.sqrt(sample_size[j]), # Achtung: t-Verteilung
        #                     avg_error[j,:] + q*np.sqrt(var_error[j,:])/np.sqrt(sample_size[j]), alpha=0.4,color=colors[j%6-5])

    plt.plot([plot_lo],[0])
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel(r"error $d_E$ in kcal/mol",fontsize=22)
    if experiment=='sample size' and formula=='N2':
        plt.ylim([0,1800])
    elif experiment=='hyperparameters' and formula=='N2':
        plt.ylim([0,2100])
    elif experiment=='sample size' and formula=='CO':
        plt.ylim([0,1800])
    elif experiment=='hyperparameters' and formula=='CO':
        plt.ylim([0,2600])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.legend(fontsize=22)
    plt.show()

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    # t-test
    for j in range(4):
        print("test",j)
        # test if variances are equal
        print(f.cdf(var_error[j,:]/var_error[j+1,:],sample_size[j]-1,sample_size[j+1]-1))
        # if variances are equal, test if averages are different with t test
        # var = ((sample_size[j]-1)*var_error[j,:] + (sample_size[j+1]-1)*var_error[j+1,:])/(sample_size[j] + sample_size[j+1] - 2)
        # test_statistic = (avg_error[j,:]-avg_error[j+1,:])/np.sqrt(var)/np.sqrt(1/sample_size[j]+1/sample_size[j+1]) # should be positive
        # print(t.cdf(test_statistic,sample_size[j] + sample_size[j+1] - 2),"\n")

    for j in range(12,17):
        print("test",j)
        # test if variances are equal
        print(f.cdf(var_error[11,:]/var_error[j,:],sample_size[11]-1,sample_size[j]-1))
        # if variances are equal, test if averages are different with t test
        var = ((sample_size[11]-1)*var_error[11,:] + (sample_size[j]-1)*var_error[j,:])/(sample_size[11] + sample_size[j] - 2)
        test_statistic = (avg_error[11,:]-avg_error[j,:])/np.sqrt(var)/np.sqrt(1/sample_size[11]+1/sample_size[j])
        print(t.cdf(test_statistic,sample_size[11] + sample_size[j] - 2),"\n")

    return


def density_comparison_bis():
    plot_lo = lo
    plot_hi = hi
    plot_increments = 200
    tests = 10
    trials = 1
    avg_error = np.zeros((tests,plot_increments))
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # enumerate(np.arange(1,3.2,0.25)): # 
        print("R =",R)

        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        E_values = mf.e_tot
        nuc = mf.energy_nuc()

        S = mol.intor('int1e_ovlp')
        h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_inv_root_i = np.linalg.inv(scipy.linalg.sqrtm(S))
        h_new = np.einsum('pr,rq->pq',S_inv_root_i,np.einsum('rs,sq->rq', h_i, S_inv_root_i))
        v_new = np.einsum('ap,pbcd->abcd',S_inv_root_i,np.einsum('bq,pqcd->pbcd',S_inv_root_i,np.einsum('cr,pqrd->pqcd',S_inv_root_i,np.einsum('ds,pqrs->pqrd', S_inv_root_i, mol.intor('int2e')))))
        h_new = torch.tensor(h_new,dtype=torch.float).reshape((1,B,B))
        v_new = torch.tensor(v_new,dtype=torch.float).reshape((1,B,B,B,B))
        positions = torch.tensor(positions,dtype=torch.float).reshape((1,M,3))

        for j in [2]: # range(4): # [3,5,6,7,8,9]: # 
            print("trial",j)
            error = torch.zeros((trials))
            for m in range(trials):
                if j==0:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_20_{}__trial_{}'.format(formula,basis,m)))
                    label='10 training points'
                elif j==1:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}__trial_{}'.format(formula,basis,m)))
                    label='20 training points'
                elif j==2:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,m)))
                    label='50 training points'
                elif j==3:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,m)))
                    label='100 training points'
                elif j==4:
                    explanation_model = xSchNet(basis,F_dissoc)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,m)))
                    label='200 training points'
                elif j==5:
                    explanation_model = xSchNet(n_SchNet=32,n_SchNOrb=32)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_features_trial_{}'.format(formula,basis,m)))
                    label='smaller feature dim.'
                elif j==6:
                    explanation_model = xSchNet(lmax=1)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_lmax_trial_{}'.format(formula,basis,m)))
                    label='less SchNOrb layers'
                elif j==7:
                    explanation_model = xSchNet(lmax=3)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_lmax_trial_{}'.format(formula,basis,m)))
                    label='more SchNOrb layers'
                elif j==8:
                    explanation_model = xSchNet(n_interactions=1)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_less_SchNet_interact_trial_{}'.format(formula,basis,m)))
                    label='less SchNet layers'
                elif j==9:
                    explanation_model = xSchNet(n_interactions=3)
                    explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_interact_trial_{}'.format(formula,basis,m)))
                    label='more SchNet layers'
                
                running_loss = 0
                with torch.no_grad():

                    F_new, e, E = explanation_model((positions,None,h_new))
                    _, U = torch.linalg.eigh(F_new)
                    
                    J = torch.einsum('npqjj->npq',torch.einsum('nri,npqrj->npqij',U[:,:,:N//2],torch.einsum('npqrs,nsj->npqrj', v_new, U[:,:,:N//2])))
                    K = torch.einsum('npjjs->nps',torch.einsum('nqi,npqjs->npijs',U[:,:,:N//2],torch.einsum('npqrs,nrj->npqjs', v_new, U[:,:,:N//2])))
                    E_ee = torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', 2*h_new + 2*J - K, U[:,:,:N//2])))
                    error[m] = E_ee[0] + nuc - E_values
            
            error = Hartree_to_eV/Hartree_to_kcal_per_mol*error
            avg_error[j,i] = torch.mean(error[error>0]).detach().numpy()

    for j in [2]: # range(4):
        plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),avg_error[j,:],label=label,lw=2)

    plt.plot([plot_lo],[0])
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel("Difference in kcal/mol",fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()

    return


def density_comparison_multiple_basis(formula):
    """
    Code used to generate the left hand side of Figure 4.13, i.e., compare the average physical error of models using different basis sets.
    """
    plot_lo = lo
    plot_hi = hi
    plot_increments = 200
    tests = 3
    trials = 50
    avg_error = np.zeros((tests,plot_increments))
    for i, R in enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # enumerate(np.arange(1,3.2,0.25)): # 
        print("R =",R)

        positions_np = [[-R/2,0,0],[R/2,0,0]]
        positions_np = list(map(np.array,positions_np))


        for j,basis in enumerate(['sto-6g','631g','def2-tzvp']):
            F_dissoc, E_dissoc = RHF_dissociation(basis,numbers)
            B, B_k, N, M = get_basis_size(basis,numbers)

            mol = gto.M(atom = [[numbers[l],positions_np[l]] for l in range(len(numbers))],
                        basis = basis,
                        unit='Angstrom')
            mf = scf.RHF(mol)
            mf.kernel()
            E_values = mf.e_tot
            nuc = mf.energy_nuc()

            S = mol.intor('int1e_ovlp')
            h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
            S_inv_root_i = np.linalg.inv(scipy.linalg.sqrtm(S)).real
            h_new = np.einsum('pr,rq->pq',S_inv_root_i,np.einsum('rs,sq->rq', h_i, S_inv_root_i))
            v_new = np.einsum('ap,pbcd->abcd',S_inv_root_i,np.einsum('bq,pqcd->pbcd',S_inv_root_i,np.einsum('cr,pqrd->pqcd',S_inv_root_i,np.einsum('ds,pqrs->pqrd', S_inv_root_i, mol.intor('int2e')))))
            h_new = torch.tensor(h_new).reshape((1,B,B))
            v_new = torch.tensor(v_new).reshape((1,B,B,B,B))
            positions = torch.tensor(positions_np).reshape((1,M,3))

            error = torch.zeros((trials))
            for m in range(trials):
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,m)))

                with torch.no_grad():
                    F_new, e, E = explanation_model((positions,None,h_new))
                    _, U = torch.linalg.eigh(F_new)
                    
                J = torch.einsum('npqjj->npq',torch.einsum('nri,npqrj->npqij',U[:,:,:N//2],torch.einsum('npqrs,nsj->npqrj', v_new, U[:,:,:N//2])))
                K = torch.einsum('npjjs->nps',torch.einsum('nqi,npqjs->npijs',U[:,:,:N//2],torch.einsum('npqrs,nrj->npqjs', v_new, U[:,:,:N//2])))
                E_ee = torch.einsum('nii->n',torch.einsum('npi,npj->nij',U[:,:,:N//2],torch.einsum('npq,nqj->npj', 2*h_new + 2*J - K, U[:,:,:N//2])))
                error[m] = E_ee[0] + nuc - E_values
                
            error = Hartree_to_eV/Hartree_to_kcal_per_mol*error
            avg_error[j,i] = torch.mean(error[error>0]).detach().numpy()
    
    labels = ['STO-6G','6-31G','def2-TZVP']
    for j,basis in enumerate(['sto-6g','631g','def2-tzvp']):
        plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),avg_error[j,:],label=labels[j],lw=2)
    plt.plot([plot_lo],[0])
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel(r"error $d_E$ in kcal/mol",fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()

    return


def density_comparison_L2(experiment):
    """
    Code used to generate right-hand-sides of Figure 4.10, 4.12, 4.18, 4.20, i.e., the average L^2-physical error according to equation (4.3). 
    The input experiment takes values 'sample size' and 'hyperparameters' depending on whether models trained on different sample sizes 
    or models with different hyperparameters should be compared.
    """
    plot_lo = lo
    plot_hi = hi
    plot_increments = 200
    colors = ['C0','C1','C3','C4','C5','C6']

    S_inv_root = torch.zeros(plot_increments,B,B,dtype=torch.double)
    h = torch.zeros(plot_increments,B,B,dtype=torch.double)
    P_ref = np.zeros((plot_increments,B,N//2))
    D_ref = np.zeros((plot_increments,B,B))
    positions_list = torch.zeros(plot_increments,M,3,dtype=torch.double)
    S4 = torch.zeros(plot_increments,B,B,B,B)
    for i, R in enumerate(np.linspace(0.7,2,200)):
        print("R =",R)

        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))
        positions_list[i,:,:] = torch.tensor(positions)

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()

        S = mol.intor('int1e_ovlp')
        h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_inv_root_i = np.linalg.inv(scipy.linalg.sqrtm(S))
        S_inv_root[i,:,:] = torch.tensor(S_inv_root_i)
        h[i,:,:] = torch.tensor(h_i)
        if formula=='N2':
            S4[i,:,:,:,:] = torch.load("{}_four_center_ovlp_{}_R_{}".format(formula,basis,round(R*1000)))
        else:
            S4[i,:,:,:,:] = torch.load("temp/{}_four_center_ovlp_{}_R_{}".format(formula,basis,round(R*1000)))
        P_ref[i,:,:] = mf.mo_coeff[:,:N//2]
        D_ref[i,:,:] = mf.mo_coeff[:,:N//2] @ mf.mo_coeff[:,:N//2].transpose()


    # do nothing
    _, U = torch.linalg.eigh(F_dissoc)
    P = (S_inv_root @ U[:,:N//2]).detach().numpy()
    D = np.einsum('npi,nqi->npq',P,P)
    error = np.sqrt(np.einsum('npq,npq->n',D,np.einsum('npqrs,nrs->npq',S4,D)) + np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D_ref)) -\
            2*np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D)))
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),error,label="no learning",lw=2,color='k')

    if experiment=='sample size':
        if formula=='CO':
            tests = range(5)
        elif formula=='N2':
            tests = range(6)
    elif experiment=='hyperparameters':
        tests = [11,12,13,14,15,16]
    for c, j in enumerate(tests): # enumerate([11,12,13,14,15,16]): # enumerate([0,1,3,12]): # 
        print("model",j)
        if experiment=='sample size':
            trials = 50
        elif experiment=='hyperparameters':
            trials = 100
        L2 = np.zeros((plot_increments,trials))
        for i in range(trials):
            if j==0:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_20_{}__trial_{}'.format(formula,basis,i)))
                label='10 data points'
            elif j==1:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}__trial_{}'.format(formula,basis,i)))
                label='20 data points'
            elif j==2:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,i)))
                label='50 data points'
            elif j==3:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i)))
                label='100 data points'
            elif j==4:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='200 data points'
            elif j==5:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_1000_{}__trial_{}'.format(formula,basis,i)))
                label='500 data points'
            elif j==6:
                explanation_model = xSchNet(basis,F_dissoc,n_SchNet=32,n_SchNOrb=32)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_features_trial_{}'.format(formula,basis,i)))
                label='smaller feature dim.'
            elif j==7:
                explanation_model = xSchNet(basis,F_dissoc,lmax=1)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_lmax_trial_{}'.format(formula,basis,i)))
                label='less SchNOrb layers'
            elif j==8:
                explanation_model = xSchNet(basis,F_dissoc,lmax=3)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_lmax_trial_{}'.format(formula,basis,i)))
                label='more SchNOrb layers'
            elif j==9:
                explanation_model = xSchNet(basis,F_dissoc,n_interactions=1)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_less_SchNet_interact_trial_{}'.format(formula,basis,i)))
                label='less SchNet layers'
            elif j==10:
                explanation_model = xSchNet(basis,F_dissoc,n_interactions=3)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_interact_trial_{}'.format(formula,basis,i)))
                label='more SchNet layers'
            elif j==11:
                explanation_model = xSchNet(basis,F_dissoc)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='baseline'
            elif j==12:
                explanation_model = xSchNet(basis,F_dissoc,n_interactions=2)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='2 SchNet layers'
            elif j==13:
                explanation_model = xSchNet(basis,F_dissoc,lmax=2)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNOrb_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label=r'$l_{\text{max}}=2$'
            elif j==14:
                explanation_model = xSchNet(basis,F_dissoc,n_SchNet=3,n_SchNOrb=3)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='feature dim. 3'
            elif j==15:
                explanation_model = xSchNet(basis,F_dissoc,n_SchNet=8,n_SchNOrb=8)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='feature dim. 8'
            elif j==16:
                explanation_model = xSchNet(basis,F_dissoc,n_SchNet=64,n_SchNOrb=64)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_feature_dim_64_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='feature dim. 64'
            elif j==17:
                explanation_model = xSchNet(basis,F_dissoc,n_SchNet=64,n_SchNOrb=64,n_interactions=2)
                explanation_model.load_state_dict(torch.load('large_xSchNet/SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='large xSchNet'
            elif j==18:
                explanation_model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
                explanation_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear__trial_{}'.format(formula,basis,i)))
                label='linear model'
        
            running_loss = 0
            with torch.no_grad():
                # for m, data in enumerate(training_loader):
                #         inputs, vtargets = data
                #         outputs = explanation_model(inputs)
                #         loss = float(torch.mean(torch.abs(outputs[-1] - vtargets))) # loss_fn(voutputs, vtargets)
                #         running_loss += loss
            
                # avg_loss = running_loss / (m + 1)
                # if avg_loss>0.0015:
                #     continue

                F_new, e, E = explanation_model((positions_list,None,h))
                _, U = torch.linalg.eigh(F_new)
                P = (S_inv_root @ U[:,:,:N//2]).detach().numpy()
                D = np.einsum('npi,nqi->npq',P,P)

                # compute ||\rho_ref - \rho||^2 for this trial
                L2[:,i] = np.sqrt(np.einsum('npq,npq->n',D,np.einsum('npqrs,nrs->npq',S4,D)) + np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D_ref)) -\
                        2*np.einsum('npq,npq->n',D_ref,np.einsum('npqrs,nrs->npq',S4,D)))

        sample_size = np.sum(L2[0,:]>0)
        print(sample_size)
        avg_L2 = np.mean(L2[:,L2[0,:]>0],axis=1)
        std_L2 = np.std(L2[:,L2[0,:]>0],axis=1)
        plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),avg_L2,label=label,lw=2,color=colors[c%6])
        # if j==3:
        #     q = t.ppf(0.975,sample_size-1)
        #     plt.fill_between(np.linspace(plot_lo,plot_hi,plot_increments),
        #                     avg_L2 - q*std_L2/np.sqrt(sample_size), # Achtung: t-Verteilung
        #                     avg_L2 + q*std_L2/np.sqrt(sample_size), alpha=0.4,color="C{}".format(j))


    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel(r"error $d_{L^2}$ in $a_0^{-3/2}$",fontsize=22)
    #plt.ylabel(r"error $||\rho_{ref}-\rho||_{L^2}$",fontsize=22)
    if experiment=='sample size' and formula=='N2':
        plt.ylim([0,0.2])
    elif experiment=='hyperparameters' and formula=='N2':
        plt.ylim([0,0.24])
    elif experiment=='sample size' and formula=='CO':
        plt.ylim([0,0.27])
    elif experiment=='hyperparameters' and formula=='CO':
        plt.ylim([0,0.29])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()
    return

def density_map():
    """
    Code used to generate Figure 4.11 and 4.19, i.e., the average density difference between a predicted density and the reference density.
    """
    for R in [R_star]: # np.arange(0.7,2.1,0.1):
        print("R =",R)

        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        D_ref = mf.mo_coeff[:,:N//2] @ mf.mo_coeff[:,:N//2].transpose()

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc')).reshape((1,B,B))
        S_root = scipy.linalg.sqrtm(S)
        S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S)).real).reshape((1,B,B))
        h_new = torch.einsum('npr,nrq->npq',S_inv_root,torch.einsum('nrs,nsq->nrq', h, S_inv_root))
        
        # compute reference density
        xmin=-2
        xmax=2
        ymin=-1
        ymax=1
        xn=200
        yn=100
        X = np.linspace(xmin,xmax,xn)
        Y = np.linspace(ymin,ymax,yn)
        positions = torch.tensor(positions).reshape((1,M,3))
        positions_np = positions.detach().numpy().reshape((M,3))

        # compute ref values
        values_ref = np.array([rho(X[i],Y[j],0,D_ref,numbers,positions_np,orbitals[numbers[0]],basis_func,B) for j in range(len(Y)) for i in range(len(X))])
        values_ref = values_ref.reshape((yn,xn))
        print("reference values done...")

        # plot ref density
        imshow_kwargs = {
                'vmax': 0.4,
                'vmin': 0,
                'cmap': 'Greys',
                'extent': (xmin, xmax, ymin, ymax),
            }

        fig, ax = plt.subplots()
        pos = ax.imshow(values_ref, **imshow_kwargs)
        for k,Z_k in enumerate(numbers):
            if Z_k == 6:
                color = 'white'
            elif Z_k == 7:
                color = 'b'
            elif Z_k == 8:
                color = 'r'
            plt.scatter([positions_np[k,0]],[0],s=100,c=color)
        #plt.xlabel(r"$x$ position in Angstrom",fontsize=10)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.tick_params(bottom=False,left=False)
        #fig.colorbar(pos)
        plt.savefig("images/{}_{}_density_ref_R_{}.png".format(formula,basis,round(R*100)))
        plt.close()

        # make orbital prediction
        for j in [20]: # range(6): # [14,15,16,17,18,19]: # 
            trials = 50
            P = np.zeros((trials,B,B))
            for i in range(trials):
                if j==0:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_20_{}__trial_{}'.format(formula,basis,i)))
                    label='10'
                elif j==1:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}__trial_{}'.format(formula,basis,i)))
                    label='20'
                elif j==2:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,i)))
                    label='50'
                elif j==3:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i)))
                    label='100'
                elif j==4:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='200'
                elif j==5:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_1000_{}__trial_{}'.format(formula,basis,i)))
                    label='500'
                elif j==6:
                    model = xSchNet(basis,F_dissoc,n_SchNet=32,n_SchNOrb=32)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_features_trial_{}'.format(formula,basis,i)))
                    label='smaller_feature'
                elif j==7:
                    model = xSchNet(basis,F_dissoc,lmax=1)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_lmax_trial_{}'.format(formula,basis,i)))
                    label='smaller_lmax'
                elif j==8:
                    model = xSchNet(basis,F_dissoc,lmax=3)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_lmax_trial_{}'.format(formula,basis,i)))
                    label='larger_lmax'
                elif j==9:
                    model = xSchNet(basis,F_dissoc,n_interactions=1)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_less_SchNet_interact_trial_{}'.format(formula,basis,i)))
                    label='less_SchNet_interact'
                elif j==10:
                    model = xSchNet(basis,F_dissoc,n_interactions=3)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_interact_trial_{}'.format(formula,basis,i)))
                    label='more_SchNet_interact'
                elif j==11:
                    model = xSchNet(basis,F_dissoc,n_interactions=2)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_2000_{}_more_SchNet_layers_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='more_SchNet_layers'
                elif j==12:
                    model = xSchNet(basis,F_dissoc,lmax=2)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_2000_{}_more_SchNOrb_layers_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='more_SchNOrb_layers'
                elif j==13:
                    model = xSchNet(basis,F_dissoc,n_SchNet=3,n_SchNOrb=3)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_2000_{}_smaller_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='smaller_feature_dim'
                elif j==14:
                    model = xSchNet(basis,F_dissoc)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='baseline'
                elif j==15:
                    model = xSchNet(basis,F_dissoc,n_interactions=2)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNet_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='2_SchNet_layers'
                elif j==16:
                    model = xSchNet(basis,F_dissoc,lmax=2)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_more_SchNOrb_layer_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='2_SchNOrb_layers'
                elif j==17:
                    model = xSchNet(basis,F_dissoc,n_SchNet=3,n_SchNOrb=3)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_smaller_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature_dim_3'
                elif j==18:
                    model = xSchNet(basis,F_dissoc,n_SchNet=8,n_SchNOrb=8)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_larger_feature_dim_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature_dim_8'
                elif j==19:
                    model = xSchNet(basis,F_dissoc,n_SchNet=64,n_SchNOrb=64)
                    model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}_feature_dim_64_trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                    label='feature_dim_64'
                elif j==20:
                    model = NoLearning_model(F_dissoc)
                    label='no_learning'
                

                F_new, e, E = model((positions,S_inv_root,h_new))
                F_new = F_new.detach().numpy().reshape((B,B))
                F = np.einsum('pr,rq->pq',S_root,np.einsum('rs,sq->rq', F_new, S_root))
                P[i,:,:] = scipy.linalg.eigh(F,S)[1]
            
            print(P.shape[0])
            D = np.einsum('npi,nqi->npq',P[:,:,:N//2],P[:,:,:N//2])
            D = np.mean(D[D[:,0,0]!=0,:,:],axis=0)

            # compute predicted density
            values = np.array([rho(X[i],Y[j],0,D,numbers,positions_np,orbitals[numbers[0]],basis_func,B) for j in range(len(Y)) for i in range(len(X))])
            print("prediction values done...")
            values = values.reshape((yn,xn))
            diff = values - values_ref

            # plot pred-ref density
            vmax = 0.05
            imshow_kwargs = {
                'vmax': vmax,
                'vmin': -vmax,
                'cmap': 'PRGn',
                'extent': (xmin, xmax, ymin, ymax),
            }

            fig, ax = plt.subplots()
            pos = ax.imshow(diff, **imshow_kwargs)
            for k,Z_k in enumerate(numbers):
                if Z_k == 6:
                    color = 'k'
                elif Z_k == 7:
                    color = 'b'
                elif Z_k == 8:
                    color = 'r'
                plt.scatter([positions_np[k,0]],[0],s=100,c=color)
            #plt.xlabel(r"$x$ position in Angstrom",fontsize=10)
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            ax.tick_params(bottom=False,left=False)
            #fig.colorbar(pos)
            plt.savefig("images/{}_{}_{}_density_diff_map_pred-ref_R_{}.png".format(formula,basis,label,round(R*100)))
            plt.close()

            # plot predicted density
            imshow_kwargs = {
                'vmax': 0.4,
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
                plt.scatter([positions_np[k,0]],[0],s=100,c=color)
            #plt.xlabel(r"$x$ position in Angstrom",fontsize=10)
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            ax.tick_params(bottom=False,left=False)
            #fig.colorbar(pos)
            plt.savefig("images/{}_{}_{}_density_pred_R_{}.png".format(formula,basis,label,round(R*100)))
            plt.close()
        
    return

def comparison_linear_model(n):
    plot_lo = lo
    plot_hi = hi
    plot_increments = 200
    E_values = np.zeros(plot_increments)
    E_pred = np.zeros(plot_increments)
    E_linear = np.zeros(plot_increments)
    density_dist = np.zeros(plot_increments)
    density_dist_linear = np.zeros(plot_increments)
    for i, R in enumerate([1.12]): # enumerate(np.linspace(plot_lo,plot_hi,plot_increments)): # enumerate(np.arange(1,3.2,0.25)): # 
        print("R =",R)

        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()
        counter = 0
        while mf.e_tot > E_dissoc and R>2 and counter<5:
            # if often not converged, then skip and don't save the energy
            counter+=1
            mf = scf.RHF(mol)
            mf.kernel()
        if counter==5:
            continue
        v = mol.intor('int2e')
        E_values[i] = mf.e_tot - mf.energy_nuc()

        S = mol.intor('int1e_ovlp')
        h = torch.tensor(mol.intor('int1e_kin') + mol.intor('int1e_nuc')).reshape((1,B,B))
        S_root = scipy.linalg.sqrtm(S)
        S_inv_root = torch.tensor(np.linalg.inv(scipy.linalg.sqrtm(S))).reshape((1,B,B))
        h_new = torch.einsum('npr,nrq->npq',S_inv_root,torch.einsum('nrs,nsq->nrq', h, S_inv_root))
        
        positions = torch.tensor(positions).reshape((1,M,3))
        model = xSchNet(basis,F_dissoc,gpu=True)
        model.load_state_dict(torch.load('SchNOrb_v8_2_{}_{}_{}_success'.format(formula,n,basis)))
        F_new, e, E = model((positions,S_inv_root,h_new))
        _, L = torch.linalg.eigh(F_new)
        #U = S_inv_root.reshape((B,B)) @ L
        F_new = F_new.detach().numpy().reshape((B,B))
        F = np.einsum('pr,rq->pq',S_root,np.einsum('rs,sq->rq', F_new, S_root))
        _, U = scipy.linalg.eigh(F,S)
        E_pred[i] = E
        J = np.einsum('pqjj->pq',np.einsum('ri,pqrj->pqij',U[:,:N//2],np.einsum('pqrs,sj->pqrj', v, U[:,:N//2])))
        K = np.einsum('pjjs->ps',np.einsum('qi,pqjs->pijs',U[:,:N//2],np.einsum('pqrs,rj->pqjs', v, U[:,:N//2])))
        E_ee = np.einsum('ii->',np.einsum('pi,pj->ij',U[:,:N//2],np.einsum('pq,qj->pj', 2*mol.intor('int1e_kin') + 2*mol.intor('int1e_nuc') + 2*J - K, U[:,:N//2])))
        density_dist[i] = (E_ee + mf.energy_nuc() - mf.e_tot)*Hartree_to_eV/Hartree_to_kcal_per_mol

        linear_model = XAI_model_linear(formula,basis,F_dissoc,idx_i,idx_j,numbers)
        linear_model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}_linear_success'.format(formula,basis)))
        F_new, e, E = linear_model((positions,S_inv_root,h_new))
        _, L = torch.linalg.eigh(F_new)
        F_new = F_new.detach().numpy().reshape((B,B))
        F = np.einsum('pr,rq->pq',S_root,np.einsum('rs,sq->rq', F_new, S_root))
        _, U = scipy.linalg.eigh(F,S)
        E_linear[i] = E
        J = np.einsum('pqjj->pq',np.einsum('ri,pqrj->pqij',U[:,:N//2],np.einsum('pqrs,sj->pqrj', v, U[:,:N//2])))
        K = np.einsum('pjjs->ps',np.einsum('qi,pqjs->pijs',U[:,:N//2],np.einsum('pqrs,rj->pqjs', v, U[:,:N//2])))
        E_ee = np.einsum('ii->',np.einsum('pi,pj->ij',U[:,:N//2],np.einsum('pq,qj->pj', 2*mol.intor('int1e_kin') + 2*mol.intor('int1e_nuc') + 2*J - K, U[:,:N//2])))
        density_dist_linear[i] = (E_ee + mf.energy_nuc() - mf.e_tot)*Hartree_to_eV/Hartree_to_kcal_per_mol

    print("MAE xSchNet:",np.mean(np.abs(E_pred-E_values))*Hartree_to_eV/Hartree_to_kcal_per_mol)
    print("MAE linear model:",np.mean(np.abs(E_linear-E_values))*Hartree_to_eV/Hartree_to_kcal_per_mol)

    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),density_dist_linear,label='linear model')
    plt.plot(np.linspace(plot_lo,plot_hi,plot_increments),density_dist,label='xSchNet')
    plt.xlabel(r"Distance $R=|\mathbf{R}_1-\mathbf{R}_2|$ in Angstrom",fontsize=22)
    plt.ylabel("Distance in kcal/mol",fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()    

    return


def comparison_sample_size_performance(formula):
    """
    Code used to generate Figure 4.8 and 4.16.
    """
    lo = 0.7
    hi = 2
    increments = 200

    target_E = torch.zeros(increments)
    inputs = []
    F = torch.zeros((increments,B,B),dtype=torch.double)
    h = torch.zeros((increments,B,B),dtype=torch.double)
    S = torch.zeros((increments,B,B),dtype=torch.double)
    for i, R in enumerate((hi-lo)*np.random.rand(increments)+lo):
        print("R =",R)
        positions = [[-R/2,0,0],[R/2,0,0]]
        positions = list(map(np.array,positions))

        mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                    basis = basis,
                    unit='Angstrom')
        mf = scf.RHF(mol)
        mf.kernel()

        h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        S_i = mol.intor('int1e_ovlp')
        F_i = mf.get_fock()
        F[i,:,:] = torch.tensor(F_i)
        h[i,:,:] = torch.tensor(h_i)
        S[i,:,:] = torch.tensor(S_i)
        inputs.append(torch.tensor(positions))
        target_E[i] = mf.e_tot - mf.energy_nuc()

    # create data loader
    targets = torch.stack((F,h,S))
    data   = SchNet_XAIDataSet(targets, target_E, inputs)
    data_loader = torch.utils.data.DataLoader(data,batch_size=batch_size)

    # evaluate models
    losses = np.zeros(6)
    std_losses = np.zeros(6)
    for j in range(6):
        trials = 50
        loss_j = np.zeros(trials)
        for i in range(trials):
            if j==0:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_20_{}__trial_{}'.format(formula,basis,i)))
                label='10'
            elif j==1:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_40_{}__trial_{}'.format(formula,basis,i)))
                label='20'
            elif j==2:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_100_{}__trial_{}'.format(formula,basis,i)))
                label='50'
            elif j==3:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_200_{}__trial_{}'.format(formula,basis,i)))
                label='100'
            elif j==4:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_400_{}__trial_{}'.format(formula,basis,i),map_location=torch.device('cpu')))
                label='200'
            elif j==5:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_1000_{}__trial_{}'.format(formula,basis,i)))
                label='500'
            elif j==6:
                model = xSchNet(basis,F_dissoc)
                model.load_state_dict(torch.load('SchNOrb_v8_2_{}_2000_{}__trial_{}'.format(formula,basis,i)))
                label='1000'

            
            running_loss = 0
            with torch.no_grad():
                for m, data in enumerate(data_loader):
                    inputs, vtargets = data
                    outputs = model(inputs)
                    loss = float(torch.mean(torch.abs(outputs[-1] - vtargets)))
                    running_loss += loss
            
            loss_j[i] = running_loss / (m + 1)
        
        # average the loss over all trial models
        avg_loss = np.mean(loss_j[loss_j>0])
        std_loss = np.std(loss_j[loss_j>0])
        losses[j] = avg_loss*Hartree_to_eV/Hartree_to_kcal_per_mol
        std_losses[j] = std_loss*Hartree_to_eV/Hartree_to_kcal_per_mol
        print(avg_loss)
        q = t.ppf(0.975,trials-1)
        err = q*std_loss/np.sqrt(trials)
        print(err)


    plt.plot([10,20,50,100,200,500],losses,"o",markersize=10)
    for i,n in enumerate([10,20,50,100,200,500]):
        q = t.ppf(0.975,trials-1)
        err = q*std_losses[i]/np.sqrt(trials)
        plt.errorbar(n,losses[i],yerr=err, color="k", capsize=4.0)
    plt.xlabel("sample size",fontsize=22)
    plt.xscale('log')
    plt.xticks([10,20,50,100,200,500],[10,20,50,100,200,500],fontsize=22)
    plt.ylabel("average MAE in kcal/mol",fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    return


def compute_ref_values(i,j,positions,R_increments):
    """
    Computing reference values and saving them in appropriate arrays.
    """
    mol = gto.M(atom = [[numbers[l],positions[l]] for l in range(len(numbers))],
                basis = basis,
                unit='Angstrom')
    mf = scf.RHF(mol)
    mf.kernel()

    h_i = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    S_i = mol.intor('int1e_ovlp')
    F_i = mf.get_fock()
    F[i+R_increments*j,:,:] = torch.tensor(F_i)
    h[i+R_increments*j,:,:] = torch.tensor(h_i)
    S[i+R_increments*j,:,:] = torch.tensor(S_i)
    inputs.append(torch.tensor(positions))
    
    target_E[i+R_increments*j] = mf.e_tot




'''
main
'''

# learning parameters
formula = 'N2'
basis = '631g'
increments = 100 # number of points to sample (train + test)
mode = 'train'
load_model = False
loc = False
batch_size = 512
EPOCHS = 3000

# get molecule
if formula == "O2":
    lo = 0.7
    hi = 2
    R_star = 1.21
    numbers = np.array([8,8])
    positions = [[0,0,0],[R_star,0,0]]
    positions = list(map(np.array,positions))
elif formula == "CO":
    lo = 0.7
    hi = 2 # QM solver doesn't converge for values larger 2
    R_star = 1.12
    numbers = np.array([8,6])
    positions = [[0,0,0],[R_star,0,0]]
    positions = list(map(np.array,positions))
elif formula == "N2":
    lo = 0.7
    hi = 2
    R_star = 1.0977
    numbers = np.array([7,7])
    positions = [[0,0,0],[R_star,0,0]]
    positions = list(map(np.array,positions))
elif formula == "H2":
    lo = 0.4
    hi = 1.3
    R_star = 0.74
    numbers = np.array([1,1])
    positions = [[0,0,0],[R_star,0,0]]
    positions = list(map(np.array,positions))


# compute QC reference energies
B, B_k, N, M = get_basis_size(basis,numbers)
F_dissoc, E_dissoc = RHF_dissociation(formula,basis,numbers,positions)

target_E = torch.zeros(increments)
inputs = []
h = torch.zeros((increments,B,B),dtype=torch.double)
grad_h = torch.zeros((increments,B,B))
F = torch.zeros((increments,B,B),dtype=torch.double)
S = torch.zeros((increments,B,B),dtype=torch.double)

basis_func, orbitals = get_GTO_coeff(basis,numbers)

data = np.linspace(lo,hi,increments) # np.random.rand(increments)*(hi-lo) + lo # 
for i,R in enumerate(data):
    positions = [[0,0,0],[R,0,0]]
    positions = list(map(np.array,positions))
    compute_ref_values(i,0,positions,increments)


targets = torch.stack((F,h,S))
converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=10))
input = converter(Atoms(numbers=numbers, positions=positions))
atomic_numbers = input["_atomic_numbers"]
idx_i = input["_idx_i"]
idx_j = input["_idx_j"]

# create data loaders
indices = np.arange(increments)
training_indices = np.random.choice(indices,int(0.9*increments),replace=False)
validation_indices = np.setdiff1d(np.arange(increments),training_indices) # np.random.choice(np.setdiff1d(np.arange(increments),training_indices),int(0.1*increments),replace=False)
training_data   = SchNet_XAIDataSet(targets[:,training_indices,:,:], target_E[training_indices], [inputs[i] for i in training_indices])
validation_data = SchNet_XAIDataSet(targets[:,validation_indices,:,:], target_E[validation_indices], [inputs[i] for i in validation_indices])
training_loader = torch.utils.data.DataLoader(training_data,batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size)

# create explanation model, loss and optimizer
model = xSchNet(basis,F_dissoc,n_SchNet=10,n_SchNOrb=10)
if load_model:
    model.load_state_dict(torch.load('models/xSchNet_v2_{}_{}_{}_{}_{}_{}_{}_{}_{}_success'.format(formula,increments,basis,model.n_interactions,model.n_SchNet,model.cut,model.radial_basis,model.cutoff_fn_name,model.n_SchNOrb),map_location=torch.device('cpu')))
loss_fn = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002) 

print("number of parameters:",number_of_parameters(model))
if mode=='train':
    train(model)
elif mode=='eval':
    test_diatomics(model,basis,numbers,increments,training_indices)
elif mode=='plot':
    diatomics_orbital_plot(model,'xSchNet_v2',formula,basis,numbers,loc)

#extrapolation_plots()
#extrapolation_density_plots()
#interpolation_plots()
#compare_bond_orders()
#density_comparison_single_model()
#density_comparison('hyperparameters')
#density_comparison_multiple_basis(formula)
#density_comparison_L2('hyperparameters')
#density_map()
#comparison_linear_model(n)
#compute_four_center_overlap_tensor(formula,basis,numbers,orbitals[numbers[0]],basis_func)
#comparison_sample_size_performance(formula)


