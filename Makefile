CC=g++
OBJS=test.o svm.o svm_tree.o 
OBJS_MAIN=main.o svm.o svm_tree.o 
ARGS=-g -std=c++0x
LINKARGS= -lgtest_main -lgtest -pthread
LINKARGS_MAIN=  -pthread
all: test

test: $(OBJS)
	$(CC) $(ARGS) $(LINKARGS) -o $@ $^

code_main: $(OBJS_MAIN)
	$(CC) $(ARGS) $(LINKARGS_MAIN) -o $@ $^

%.o:%.cpp
	$(CC) -c $(ARGS) $(INCLUDES) $+ $(OPT)

check-syntax:
	gcc -o nul -S ${CHK_SOURCES}

clean:
	rm -rf test *.o	
