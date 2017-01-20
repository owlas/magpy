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
	override CXXFLAGS+=--std=c++11 -W -Wall -pedantic -pthread -O3 -g -fopenmp -simd -qopenmp -xHost -DVERSION=\"$(GIT_VERSION)\" -DUSEMKL -DMKL_ILP64 -I$(MKLROOT)/include
	LDLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else
	override CXXFLAGS+=--std=c++11 -W -Wall -pedantic -pthread -O3 -g -fopenmp -DVERSION=\"$(GIT_VERSION)\"
	LDLIBS=-llapacke -lblas
endif

SOURCES=$(wildcard $(LIB_PATH)/*.cpp)
OBJ_FILES=$(addprefix $(OBJ_PATH)/,$(notdir $(SOURCES:.cpp=.o)))

# Default target builds the main CLI for the MOMA simulation package
default: main

main: src/main.cpp $(OBJ_FILES)
	$(CXX) 	$(CXXFLAGS) $(LDFLAGS) $< \
		$(OBJ_FILES) \
		-o $@ $(LDLIBS)

# Build the individual object files
$(OBJ_PATH)/%.o: $(LIB_PATH)/%.cpp
	$(CXX)	$(CXXFLAGS) $(LDFLAGS) -c \
		-o $@ $< \
		$(LDLIBS)

# Run the entire testing suite (long run time)
run-full-tests: test-suite run-tests
	cd test && ./convergence configs/convergence.json
	cd test && python plotting/convergence.py
	cd test && ./equilibrium configs/equilibrium.json
	cd test && python plotting/equilibrium.py

# Run the unit tests only
run-tests: test/tests
	cd test && ./tests

# Build full testing-suite
test-suite: test/tests test/convergence test/equilibrium

# The unit tests are run using googletest
test/tests: test/tests.cpp $(OBJ_FILES) $(GTEST_HEADERS) test/gtest_main.a
	$(CXX) 	$(GTEST_FLAGS) $(CXXFLAGS) $(LDFLAGS) $< test/gtest_main.a \
		$(OBJ_FILES) \
		-o $@ $(LDLIBS)

# Additional test-suit tests
test/convergence: test/convergence.cpp $(OBJ_FILES)
	$(CXX) 	$(CXXFLAGS) $(LDFLAGS) $< \
		$(OBJ_FILES) \
		-o $@ $(LDLIBS)

test/equilibrium: test/equilibrium.cpp $(OBJ_FILES)
	$(CXX) 	$(CXXFLAGS) $(LDFLAGS) $< \
		$(OBJ_FILES) \
		-o $@ $(LDLIBS)

# Builds the gtest testing suite
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
		$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

test/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) 	$(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
		-o $@ \
		$(GTEST_DIR)/src/gtest-all.cc

test/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) 	$(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
		-o $@ \
		$(GTEST_DIR)/src/gtest_main.cc

test/gtest.a : test/gtest-all.o
	$(AR) 	$(ARFLAGS) $@ $^

test/gtest_main.a : test/gtest-all.o test/gtest_main.o
	$(AR) 	$(ARFLAGS) $@ $^

clean:
	rm -f objects/*
	rm -rf *.dSYM

clean-tests:
	rm -f test.out*
	rm -f convergence.*
	rm -f test/output/*
	rm -f test/*.o test/*.a

clean-all: clean clean-tests
