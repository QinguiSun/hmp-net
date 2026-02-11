
======= GENERAL INFORMATION =======

This dataset probes chemical reactions of methyl halides with halide anions, i.e. 
X- + CH3Y -> CH3X +  Y-, and contains structures for all possible combinations of 
X,Y = F, Cl, Br, I. The dataset also includes various structures for several smaller 
molecules that can be formed in fragmentation reactions, such as CH3X, HX, CHX or 
CH2X- as well as geometries for H2, CH2, CH3+ and XY interhalogen compounds. In total, 
the dataset provides reference energies, forces, and dipole moments for 452709 structures 
calculated at the DSD-BLYP-D3(BJ)/def2-TZVP level of theory [1-4] using the ORCA 4.0.1 
code [5,6]. 

For more details, see https://arxiv.org/abs/1902.08408.

[1] Grimme, S.; Antony, J.; Ehrlich, S. and Krieg, H. J. Chem. Phys. 132, 154104 (2010).
[2] Kozuch, S.; Gruzman, D. and Martin, J. M. J. Phys. Chem. C 114, 20801-20808 (2010).
[3] Grimme, S.; Ehrlich, S. and Goerigk, L. J. Comput. Chem. 32, 1456-1465 (2011).
[4] Weigend, F. and Ahlrichs, R. Phys. Chem. Chem. Phys. 7, 3297-3305 (2005).
[5] Neese, F. Wiley Interdiscip. Rev. Comput. Mol. Sci. 2, 73-78 (2012).
[6] Neese, F. Wiley Interdiscip. Rev. Comput. Mol. Sci. 8, e1327 (2018).



======= HOW TO CITE? =======

When using this dataset, please cite the following paper:
Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).

and the digital object identifier (DOI):
Unke, O.T. and Meuwly, M. (2019). SN2 reactions dataset. Zenodo. http://doi.org/10.5281/zenodo.2605341.



======= DATA FORMAT =======

The dataset is stored as python dictionary in a compressed numpy binary file (.npz). The dictionary contains seven numpy arrays:

R (num_data, max_atoms, 3): Cartesian coordinates of nuclei (in Angstrom [A])
Q (num_data,):              Total charge (in elementary charges [e])
D (num_data, 3):            Dipole moment vector with respect to the origin (in elementary charges times Angstrom [eA])
E (num_data,):              Potential energy with respect to free atoms (in electronvolt [eV])
F (num_data, max_atoms, 3): Forces acting on the nuclei (in electronvolt per Angstrom [eV/A])
Z (num_data, max_atoms):    Nuclear charges/atomic numbers of nuclei
N (num_data,):              Number of atoms in each structure (structures consisting of less than max_atoms entries are zero-padded)


Please note that the potential energy is given with respect to free atoms (i.e. total atomization). 
The following constants were subtracted from the original values for each occurence of the corresponding elements:

H : -13.579407869766147 eV
C : -1028.9362774711024 eV
F : -2715.578463075019 eV
Cl: -12518.663203367176 eV
Br: -70031.09203874589 eV
I : -8096.587166328217 eV

In order to recover the original values, simply add the constants back. 


To read the dataset, load the dictionary with python:

>>> data = np.load("sn2_reactions.npz")

and access individual entries with the appropriate dictionary key, e.g. "Z" for the nuclear charges:

>>> nuclear_charges = data["Z"]

See also "read_data.py" for a more comprehensive example.
