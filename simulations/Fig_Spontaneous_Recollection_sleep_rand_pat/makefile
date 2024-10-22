CC=g++
SRC := $(shell pwd)/src
INCLUDE_DIR := $(shell pwd)/include/
CFLAGS=-pthread -g -Wall -std=c++17

all : main

main: obj $(SRC)/simulation.cc $(SRC)/utils.cc $(SRC)/network.cc $(SRC)/timer.cc obj/network.o obj/utils.o obj/timer.o
	$(CC) $(CFLAGS) $(SRC)/simulation.cc -o main -I$(INCLUDE_DIR) ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/network.o : $(SRC)/network.cc
	$(CC) -c $(CFLAGS) $(SRC)/network.cc -o obj/network.o -I$(INCLUDE_DIR)

obj/utils.o : $(SRC)/utils.cc
	$(CC) -c $(CFLAGS) $(SRC)/utils.cc -o obj/utils.o -I$(INCLUDE_DIR)

obj/timer.o : $(SRC)/timer.cc
	$(CC) -c $(CFLAGS) $(SRC)/timer.cc -o obj/timer.o -I$(INCLUDE_DIR)