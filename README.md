minimum_probability_flow
========================

Minimum Probability Flow (http://arxiv.org/abs/0906.4779) implemented in python, using theano.

Optimization options:
- L-BFGS 		(scipy.optimize)
- SOF			(https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer).


Notes: 
- SOF optimizer doesn't seem to be working properly here yet.
- scipy.optimize seems to require double precision, while gpu only handles single precision.  This seems to break the optimization right now, if the code is run on the gpu (THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python src/mpf.py)


