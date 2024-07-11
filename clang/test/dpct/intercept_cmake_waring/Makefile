program = main


main: foo.cpp
	mkdir -p build
	cd build && cmake .. && make

clean:
	rm -rf build foo.o

run: $(program)
	./$(program)
