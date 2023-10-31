program = main


main: foo.cu
	nvcc -c foo.cu
	mkdir -p build
	cd build && cmake .. && make

clean:
	rm -rf build foo.o

run: $(program)
	./$(program)
