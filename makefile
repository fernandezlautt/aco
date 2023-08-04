CC = nvcc

CCFLAGS = -arch=all 

aco : bin/aco.o bin/main.o bin/graph.o bin/system.o bin/util.o bin/utilCuda.o
	$(CC) $(CCFLAGS) -o bin/aco bin/aco.o bin/main.o bin/graph.o bin/system.o bin/util.o bin/utilCuda.o 

bin/aco.o : src/aco.cu src/aco.cuh src/graph.cuh src/util.cuh src/utilCuda.cuh
	$(CC) $(CCFLAGS) -c src/aco.cu -o bin/aco.o

bin/main.o : src/main.cu src/aco.cuh src/graph.cuh src/system.cuh
	$(CC) $(CCFLAGS) -c src/main.cu -o bin/main.o

bin/graph.o : src/graph.cu src/graph.cuh
	$(CC) $(CCFLAGS) -c src/graph.cu -o bin/graph.o

bin/system.o : src/system.cu src/system.cuh src/graph.cuh
	$(CC) $(CCFLAGS) -c src/system.cu -o bin/system.o

bin/util.o : src/util.cu src/util.cuh src/system.cuh
	$(CC) $(CCFLAGS) -c src/util.cu -o bin/util.o

bin/utilCuda.o : src/utilCuda.cu src/utilCuda.cuh 
	$(CC) $(CCFLAGS) -c src/utilCuda.cu -o bin/utilCuda.o

clean:
	rm -f bin/*.o bin/aco