lightning:
	g++ -shared -w -std=c++14 -I${CONDA_PREFIX}/include/ -fPIC `python3 -m pybind11 --includes` lightning_qubit_ops.cpp -o lightning_qubit_ops -lblas -llapack -DHAVE_CBLAS=1
