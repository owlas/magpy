# makefile
#
#    make - defaults to main
#    make tests - builds tests
#    make clean - cleans up


CXX=icpc
INC_PATH=include
OBJ_PATH=objects
LIB_PATH=lib
SRC_PATH=src

GTEST_DIR=googletest
TESTS=test/test.cpp
TEST_LIB=libgtest.a
GTEST_FLAGS=-isystem $(GTEST_DIR)/include

GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)

ifeq ($(CXX),icpc)
	CXXFLAGS=--std=c++11 -W -Wall -pedantic -pthread -O3 -g -fopenmp -simd -qopenmp -xHost -DVERSION=\"$(GIT_VERSION)\"
else
	CXXFLAGS=--std=c++11 -W -Wall -pedantic -pthread -O3 -g -fopenmp -DVERSION=\"$(GIT_VERSION)\"
endif

LDFLAGS=-llapacke -lblas

SOURCES=$(wildcard $(LIB_PATH)/*.cpp)
OBJ_FILES=$(addprefix $(OBJ_PATH)/,$(notdir $(SOURCES:.cpp=.o)))

# Default target builds the main CLI for the MOMA simulation package
default: main

main: src/main.cpp $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $< \
	$(OBJ_FILES) $(LDFLAGS) \
	-o $@

# Build the individual object files
$(OBJ_PATH)/%.o: $(LIB_PATH)/%.cpp
	$(CXX) 	$(CXXFLAGS) -c \
	-o $@ $<
	$(LDFLAGS)

# The tests are run using googletest
tests: test/tests.cpp $(OBJ_FILES) $(GTEST_HEADERS) gtest_main.a
	$(CXX) $(GTEST_FLAGS) $(CXXFLAGS) $< gtest_main.a \
	$(OBJ_FILES) $(LDFLAGS) \
	-o $@

# Builds the gtest testing suite
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
		$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	$(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	$(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -f objects/*
	rm -f gtest-all.o gtest_main.o
	rm -rf *.dSYM

clean-tests:
	rm -f test.out*

clean-all: clean clean-tests
	rm -f main tests
	rm -f gtest_main.a gtest.a
