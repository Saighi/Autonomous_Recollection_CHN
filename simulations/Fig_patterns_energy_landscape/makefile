CC=g++ 
SRC := $(shell pwd)/src
INCLUDE_DIR := $(shell pwd)/include/
CFLAGS=-pthread -g -Wall -std=c++17 -O3 -march=native

all : main

main: obj $(SRC)/simulations.cc $(SRC)/utils.cc $(SRC)/network.cc obj/network.o obj/utils.o 
	$(CC) $(CFLAGS) $(SRC)/simulations.cc -o simulations -I$(INCLUDE_DIR) ./obj/*.o

obj: 
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/network.o : $(SRC)/network.cc
	$(CC) -c $(CFLAGS) $(SRC)/network.cc -o obj/network.o -I$(INCLUDE_DIR)

obj/utils.o : $(SRC)/utils.cc
	$(CC) -c $(CFLAGS) $(SRC)/utils.cc -o obj/utils.o -I$(INCLUDE_DIR)

run: simulations
	./simulations