CC = gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -fopenmp
LDFLAGS ?= -fopenmp -lm

OBJS = ../core_c/morton.o ../core_c/tree.o ../core_c/traverse.o ../core_c/integrate.o ../core_c/io.o

.PHONY: all clean

all: bh

bh: main.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ main.o $(OBJS) $(LDFLAGS)

main.o: main.c ../core_c/common.h ../core_c/io.h ../core_c/tree.h
	$(CC) $(CFLAGS) -c -o $@ main.c

../core_c/%.o: ../core_c/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f bh main.o $(OBJS)
