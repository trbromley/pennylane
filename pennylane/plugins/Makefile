.DEFAULT_GOAL := lightning_qubit

XTENSOR_INC = -D/home/tom/miniconda3/envs/pennylane/include/xtensor-blas/
XTENSOR_FLAGS = -DXTENSOR_USE_XSIMD -lblas -llapack

init:
	mkdir -p build
	cd build; conan install ..

compile:
	cd build; cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release $(XTENSOR_INC) $(XTENSOR_FLAGS)
	cd build; cmake --build .

move:
	mv build/bin/lqo .

lightning_qubit:
	mkdir -p build
	cd build; conan install ..
	cd build; cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
	cd build; cmake --build .
	mv build/bin/lqo .
