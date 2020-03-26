.DEFAULT_GOAL := lightning_qubit

init:
	mkdir -p build
	cd build; conan install ..

compile:
	cd build; cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
	cd build; cmake --build .

move:
	mv build/bin/lqo .

lightning_qubit:
	mkdir -p build
	cd build; conan install ..
	cd build; cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
	cd build; cmake --build .
	mv build/bin/lqo .
