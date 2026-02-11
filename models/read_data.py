import numpy as np

# read dataset into memory
data = np.load("sn2_reactions.npz")

# print contents to console
print("Contents:")
for key in data.keys():
    print(key, data[key].shape)
print()

# as example for accessing individual entries, print the data for entry idx=0
idx = 0
print("Data for entry " + str(idx)+":")
print("Number of atoms")
print(data["N"][idx])
print("Energy [eV]")
print(data["E"][idx])
print("Total charge")
print(data["Q"][idx])
print("Dipole moment vector (with respect to [0.0 0.0 0.0]) [eA]")
print(data["D"][idx,:])
print("Nuclear charges")
print(data["Z"][idx,:data["N"][idx]])
print("Cartesian coordinates [A]")
print(data["R"][idx,:data["N"][idx],:])
print("Forces [eV/A]")
print(data["F"][idx,:data["N"][idx],:])