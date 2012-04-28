CC=g++
OBJS=test.o svm.o svm_tree.o 
OBJS_MAIN=main.o svm.o svm_tree.o 
ARGS=-O2 -std=c++0x
LINKARGS= -pthread
GTEST_DIR = /home/jiang/lib/gtest-1.6.0
LINKARGS_MAIN=  -pthread
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)


all: test code_main

test: $(OBJS) 
	$(CC) $(ARGS) gtest_main.a $(LIBDIR) $(LINKARGS)  -o $@ $^

code_main: $(OBJS_MAIN)
	$(CC) $(ARGS) $(LINKARGS_MAIN) -o $@ $^

%.o:%.cpp
	$(CC) -c $(ARGS) $(INCLUDES) $+ $(OPT)

check-syntax:
	$(CC) $(ARGS) -o nul -S ${CHK_SOURCES}

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -rf test *.o	
