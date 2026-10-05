# 0002. Store weights as plain arrays in an `.npz` file, never as a pickle

- Status: accepted
- Date: 2026-10-05

## Context

Before version 1.0.0 the trainer wrote the weights with `pickle.dumps` and the evaluation read them with `pickle.loads`. Unpickling runs whatever code the file asks for, so a weights file handed from one person to another was a program, not data. The weights are six floating-point arrays; nothing about them needs pickle.

Two alternatives were weighed. Keeping pickle and documenting the risk would have preserved old files, but the mitigation is "only load files you wrote", which is no mitigation for a project whose users are told to share results. Writing six `.npy` files would have been as safe, but one file is easier to pass around and to name on the command line.

## Decision

Weights are written with `numpy.savez` to one `.npz` archive holding `dense1_weights`, `dense1_biases`, `dense2_weights`, `dense2_biases`, `dense3_weights` and `dense3_biases`, and read with `numpy.load(..., allow_pickle=False)`. `Network.load` checks, in this order: the file exists; the path ends in `.npz`; the archive holds exactly those six arrays; every array's header declares the shape of its layer and a floating-point type, read before any array data; every value is finite. Because the shapes are checked from the headers, a file cannot make the loader allocate more memory than the network itself occupies. The arrays are converted to 64-bit floats, the type the network trains in. A failed check raises `WeightsError` with the reason and leaves the network as it was. A path that does not end in `.npz` is refused without being opened.

The file is written under a `.part` name and renamed, so an interrupted run never leaves a truncated file under the real name.

## Consequences

- Loading weights cannot execute code from the file. An object array inside an archive is refused, not unpickled.
- Pickle checkpoints written by the scripts before 1.0.0 (`mnist_weights.pkl`) no longer load, and there is no converter, because a converter would have to unpickle. Retraining takes a minute or two and, in the locked environment, gives the same weights.
- The archive stores a timestamp, so two files with identical weights differ as files. `Network.fingerprint`, a SHA-256 over the six arrays, is the way to compare weights, and both commands print it.
- The file holds parameters only. Resuming a run would need the optimiser's moment buffers and step count as well; that is out of scope until someone needs it, and would be a new file format with its own record.
