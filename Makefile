all:
	g++ -I./include -o bin/main.o -c src/main.cpp
	g++ -I./include -o bin/matrices.o -c src/matrices.cpp
	g++ -o main bin/*.o
