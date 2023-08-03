CC = gcc

CCFLAGS = -Wall -g 

aco : bin/aco.o bin/main.o bin/graph.o bin/system.o bin/util.o
	$(CC) $(CCFLAGS) -o bin/aco bin/aco.o bin/main.o bin/graph.o bin/system.o bin/util.o 

bin/aco.o : src/aco.c src/aco.h src/graph.h src/util.h
	$(CC) $(CCFLAGS) -c src/aco.c -o bin/aco.o

bin/main.o : src/main.c src/aco.h src/graph.h src/system.h
	$(CC) $(CCFLAGS) -c src/main.c -o bin/main.o

bin/graph.o : src/graph.c src/graph.h
	$(CC) $(CCFLAGS) -c src/graph.c -o bin/graph.o

bin/system.o : src/system.c src/system.h src/graph.h
	$(CC) $(CCFLAGS) -c src/system.c -o bin/system.o

bin/util.o : src/util.c src/util.h
	$(CC) $(CCFLAGS) -c src/util.c -o bin/util.o

clean:
	rm -f bin/*.o bin/aco