import h5py
f = h5py.File('elmo_test.hdf5', 'r')
print(list(f.keys()))
