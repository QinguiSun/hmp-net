import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.RBFLayer import *
from .layers.InteractionBlock import *
from .layers.OutputBlock      import *
from .layers.activation_fn import *
from .grimme_d3.grimme_d3 import *
import numpy as np

def softplus_inverse(x):
    '''numerically stable inverse of softplus transform'''
    return x + np.log(-np.expm1(-x))

class PhysNetmodel(nn.Module):
    def __str__(self):
        return "PhysNet Neural Network"

    def __init__(self,
                 F,                              #dimensionality of feature vector
                 K,                              #number of radial basis functions
                 sr_cut,                         #cutoff distance for short range interactions
                 lr_cut = None,                  #cutoff distance for long range interactions (default: no cutoff)
                 num_blocks=3,                   #number of building blocks to be stacked
                 num_residual_atomic=2,          #number of residual layers for atomic refinements of feature vector
                 num_residual_interaction=2,     #number of residual layers for refinement of message vector
                 num_residual_output=1,          #number of residual layers for the output blocks
                 use_electrostatic=True,         #adds electrostatic contributions to atomic energy
                 use_dispersion=True,            #adds dispersion contributions to atomic energy
                 s6=None,                        #s6 coefficient for d3 dispersion, by default is learned
                 s8=None,                        #s8 coefficient for d3 dispersion, by default is learned
                 a1=None,                        #a1 coefficient for d3 dispersion, by default is learned
                 a2=None,                        #a2 coefficient for d3 dispersion, by default is learned   
                 Eshift=0.0,                     #initial value for output energy shift (makes convergence faster)
                 Escale=1.0,                     #initial value for output energy scale (makes convergence faster)
                 Qshift=0.0,                     #initial value for output charge shift 
                 Qscale=1.0,                     #initial value for output charge scale 
                 kehalf=7.199822675975274,       #half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
                 activation_fn=shifted_softplus, #activation function
                 dtype=torch.float32,            #single or double precision
                 keep_prob=1.0,                 #keep probability for dropout regularization
                 seed=None,
                 scope=None):
        super(PhysNetmodel, self).__init__()
        assert(num_blocks > 0)
        self._num_blocks = num_blocks
        self._dtype = dtype
        self._kehalf = kehalf
        self._F = F
        self._K = K
        self._sr_cut = sr_cut #cutoff for neural network interactions
        self._lr_cut = lr_cut #cutoff for long-range interactions
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion
        self._activation_fn = activation_fn
        self._scope = scope

        if seed is not None:
            torch.manual_seed(seed)

        #keep probability for dropout regularization
        self.register_buffer('keep_prob', torch.tensor(keep_prob))

        #atom embeddings (we go up to Pu(94), 95 because indices start with 0)
        self._embeddings = nn.Parameter(torch.empty(95, self.F, dtype=dtype).uniform_(-np.sqrt(3), np.sqrt(3)))
        
        #radial basis function expansion layer
        self._rbf_layer = RBFLayer(K, sr_cut) # scope removed, handled by module

        #initialize variables for d3 dispersion (the way this is done, positive values are guaranteed)
        # Store raw parameters, apply softplus in property getter to match TF logic
        if s6 is None:
            self._s6_param = nn.Parameter(torch.tensor(softplus_inverse(d3_s6), dtype=dtype))
            self._s6_trainable = True
        else:
            self.register_buffer('_s6_param', torch.tensor(s6, dtype=dtype))
            self._s6_trainable = False

        if s8 is None:
            self._s8_param = nn.Parameter(torch.tensor(softplus_inverse(d3_s8), dtype=dtype))
            self._s8_trainable = True
        else:
            self.register_buffer('_s8_param', torch.tensor(s8, dtype=dtype))
            self._s8_trainable = False

        if a1 is None:
            self._a1_param = nn.Parameter(torch.tensor(softplus_inverse(d3_a1), dtype=dtype))
            self._a1_trainable = True
        else:
            self.register_buffer('_a1_param', torch.tensor(a1, dtype=dtype))
            self._a1_trainable = False
            
        if a2 is None:
            self._a2_param = nn.Parameter(torch.tensor(softplus_inverse(d3_a2), dtype=dtype))
            self._a2_trainable = True
        else:
            self.register_buffer('_a2_param', torch.tensor(a2, dtype=dtype))
            self._a2_trainable = False

        #initialize output scale/shift variables
        self._Eshift = nn.Parameter(torch.full((95,), Eshift, dtype=dtype))
        self._Escale = nn.Parameter(torch.full((95,), Escale, dtype=dtype))
        self._Qshift = nn.Parameter(torch.full((95,), Qshift, dtype=dtype))
        self._Qscale = nn.Parameter(torch.full((95,), Qscale, dtype=dtype))

        #embedding blocks and output layers
        self.interaction_block = nn.ModuleList()
        self.output_block = nn.ModuleList()
        for i in range(num_blocks):
            self.interaction_block.append(
                InteractionBlock(K, F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, seed=seed, keep_prob=self.keep_prob, dtype=dtype))
            self.output_block.append(
                OutputBlock(F, num_residual_output, activation_fn=activation_fn, seed=seed, keep_prob=self.keep_prob, dtype=dtype))
                            
    def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
        #calculate interatomic distances
        Ri = R[idx_i]
        Rj = R[idx_j]
        if offsets is not None:
            Rj += offsets
        # relu prevents negative numbers in sqrt (though theoretically impossible for dist^2)
        Dij = torch.sqrt(F.relu(torch.sum((Ri-Rj)**2, -1))) 
        return Dij

    #calculates the atomic energies, charges and distances (needed if unscaled charges are wanted e.g. for loss function)
    def atomic_properties(self, Z, R, idx_i, idx_j, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        #calculate distances (for long range interaction)
        Dij_lr = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)
        #optionally, it is possible to calculate separate distances for short range interactions (computational efficiency)
        if sr_idx_i is not None and sr_idx_j is not None:
            Dij_sr = self.calculate_interatomic_distances(R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
        else:
            sr_idx_i = idx_i
            sr_idx_j = idx_j
            Dij_sr = Dij_lr

        #calculate radial basis function expansion
        rbf = self.rbf_layer(Dij_sr)

        #initialize feature vectors according to embeddings for nuclear charges
        x = self.embeddings[Z]

        #apply blocks
        Ea = 0 #atomic energy 
        Qa = 0 #atomic charge
        nhloss = 0 #non-hierarchicality loss
        lastout2 = None
        
        for i in range(self.num_blocks):
            x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
            out = self.output_block[i](x)
            Ea += out[:,0]
            Qa += out[:,1]
            #compute non-hierarchicality loss
            out2 = out**2
            if i > 0:
                nhloss += torch.mean(out2/(out2 + lastout2 + 1e-7))
            lastout2 = out2

        #apply scaling/shifting
        Ea = self.Escale[Z] * Ea + self.Eshift[Z] # last term + 0*sum(R) removed as gradients are handled by autograd
        Qa = self.Qscale[Z] * Qa + self.Qshift[Z]
        return Ea, Qa, Dij_lr, nhloss

    def _segment_sum(self, data, segment_ids, num_segments=None):
        """Helper to replicate tf.segment_sum behavior"""
        if num_segments is None:
            num_segments = segment_ids.max().item() + 1
        # Create result tensor
        # Assumes data is [N, ...] and segment_ids is [N]
        shape = list(data.shape)
        shape[0] = num_segments
        result = torch.zeros(shape, dtype=data.dtype, device=data.device)
        return result.index_add(0, segment_ids, data)

    #calculates the energy given the scaled atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_from_scaled_atomic_properties(self, Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        
        Ea_copy = Ea.clone()
        
        #add electrostatic and dispersion contribution to atomic energy
        if self.use_electrostatic:
            Ea_copy += self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j)
        if self.use_dispersion:
            # Note: edisp needs to be compatible with PyTorch tensors
            if self.lr_cut is not None:   
                Ea_copy += d3_autoev*edisp(Z, Dij/d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2, cutoff=self.lr_cut/d3_autoang)
            else:
                Ea_copy += d3_autoev*edisp(Z, Dij/d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)
        
        # Use segment_sum equivalent
        return torch.squeeze(self._segment_sum(Ea_copy, batch_seg))

    #calculates the energy and forces given the scaled atomic atomic properties
    def energy_and_forces_from_scaled_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, batch_seg=None):
        # Ensure R requires grad for force calculation
        if not R.requires_grad:
            R.requires_grad_(True)
            
        energy = self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)
        # Calculate gradients (Forces = -dE/dR)
        forces = -torch.autograd.grad(torch.sum(energy), R, create_graph=True)[0]
        return energy, forces

    #calculates the energy given the atomic properties
    def energy_from_atomic_properties(self, Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot=None, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        #scale charges such that they have the desired total charge
        Qa = self.scaled_charges(Z, Qa, Q_tot, batch_seg)
        return self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)

    #calculates the energy and force given the atomic properties
    def energy_and_forces_from_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None):
        # Ensure R requires grad
        if not R.requires_grad:
            R.requires_grad_(True)
            
        energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)
        forces = -torch.autograd.grad(torch.sum(energy), R, create_graph=True)[0]
        return energy, forces

    #calculates the total energy (including electrostatic interactions)
    def energy(self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
        energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)
        return energy 

    #calculates the total energy and forces (including electrostatic interactions)
    def energy_and_forces(self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
        energy, forces = self.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot, batch_seg)
        return energy, forces

    #returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
    def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        
        #number of atoms per batch (needed for charge scaling)
        # Equivalent to segment_sum of ones
        ones = torch.ones_like(batch_seg, dtype=self.dtype)
        Na_per_batch = self._segment_sum(ones, batch_seg)
        
        if Q_tot is None: #assume desired total charge zero if not given
            Q_tot = torch.zeros_like(Na_per_batch, dtype=self.dtype)
            
        #return scaled charges (such that they have the desired total charge)
        # gather equivalent: index select
        charge_diff_per_mol = (Q_tot - self._segment_sum(Qa, batch_seg)) / Na_per_batch
        charge_correction = charge_diff_per_mol[batch_seg]
        return Qa + charge_correction

    #switch function for electrostatic interaction (switches between shielded and unshielded electrostatic interaction)
    def _switch(self, Dij):
        cut = self.sr_cut/2
        x  = Dij/cut
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        return torch.where(Dij < cut, 6*x5-15*x4+10*x3, torch.ones_like(Dij))

    #calculates the electrostatic energy per atom 
    #for very small distances, the 1/r law is shielded to avoid singularities
    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        #gather charges
        Qi = Qa[idx_i]
        Qj = Qa[idx_j]
        #calculate variants of Dij which we need to calculate
        #the various shileded/non-shielded potentials
        DijS = torch.sqrt(Dij*Dij + 1.0) #shielded distance
        #calculate value of switching function
        switch = self._switch(Dij) #normal switch
        cswitch = 1.0-switch #complementary switch
        #calculate shielded/non-shielded potentials
        if self.lr_cut is None: #no non-bonded cutoff
            Eele_ordinary = 1.0/Dij   #ordinary electrostatic energy
            Eele_shielded = 1.0/DijS  #shielded electrostatic energy
            #combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
        else: #with non-bonded cutoff
            cut   = self.lr_cut
            cut2  = self.lr_cut*self.lr_cut
            Eele_ordinary = 1.0/Dij  +  Dij/cut2 - 2.0/cut
            Eele_shielded = 1.0/DijS + DijS/cut2 - 2.0/cut
            #combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
            Eele = torch.where(Dij <= cut, Eele, torch.zeros_like(Eele))
        
        return self._segment_sum(Eele, idx_i, num_segments=Qa.shape[0])

    #save the current model
    def save(self, path):
        torch.save(self.state_dict(), path)

    #load a model
    def restore(self, path):
        self.load_state_dict(torch.load(path))

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def Eshift(self):
        return self._Eshift

    @property
    def Escale(self):
        return self._Escale
  
    @property
    def Qshift(self):
        return self._Qshift

    @property
    def Qscale(self):
        return self._Qscale

    # D3 properties need to handle the softplus logic from initialization
    @property
    def s6(self):
        if self._s6_trainable:
            return F.softplus(self._s6_param)
        return self._s6_param

    @property
    def s8(self):
        if self._s8_trainable:
            return F.softplus(self._s8_param)
        return self._s8_param
    
    @property
    def a1(self):
        if self._a1_trainable:
            return F.softplus(self._a1_param)
        return self._a1_param

    @property
    def a2(self):
        if self._a2_trainable:
            return F.softplus(self._a2_param)
        return self._a2_param

    @property
    def use_electrostatic(self):
        return self._use_electrostatic

    @property
    def use_dispersion(self):
        return self._use_dispersion

    @property
    def kehalf(self):
        return self._kehalf

    @property
    def F(self):
        return self._F

    @property
    def K(self):
        return self._K

    @property
    def sr_cut(self):
        return self._sr_cut

    @property
    def lr_cut(self):
        return self._lr_cut
    
    @property
    def activation_fn(self):
        return self._activation_fn
    
    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def scope(self):
        return self._scope