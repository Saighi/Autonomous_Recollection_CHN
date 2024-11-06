# Compiler settings
CC=g++
CFLAGS=-pthread -g -Wall -std=c++17

# Detect OS
ifeq ($(OS),Windows_NT)
    # Windows settings
    RM=del /Q
    MKDIR=if not exist "obj" mkdir obj
    SEP=\\
    ROOT=.
else
    # Linux settings
    RM=rm -f
    MKDIR=if [ ! -d "./obj" ]; then mkdir obj; fi
    SEP=/
    ROOT=.
endif

# Paths - using relative paths instead of pwd
SRC=$(ROOT)$(SEP)src
INCLUDE_DIR=$(ROOT)$(SEP)include

# Object files
OBJS=obj$(SEP)network.o obj$(SEP)utils.o

# Main target
all: main

main: obj $(OBJS)
	$(CC) $(CFLAGS) $(SRC)$(SEP)simulation.cc -o main -I$(INCLUDE_DIR) $(OBJS)

# Create obj directory
obj:
	$(MKDIR)

# Object files compilation
obj$(SEP)network.o: $(SRC)$(SEP)network.cc
	$(CC) -c $(CFLAGS) $(SRC)$(SEP)network.cc -o obj$(SEP)network.o -I$(INCLUDE_DIR)

obj$(SEP)utils.o: $(SRC)$(SEP)utils.cc
	$(CC) -c $(CFLAGS) $(SRC)$(SEP)utils.cc -o obj$(SEP)utils.o -I$(INCLUDE_DIR)

# Clean target
clean:
ifeq ($(OS),Windows_NT)
	$(RM) obj$(SEP)*.o
	$(RM) main.exe
else
	$(RM) obj$(SEP)*.o
	$(RM) main
endif

.PHONY: all clean obj