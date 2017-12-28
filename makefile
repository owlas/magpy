# Magpy C++ makefile
#
#    make - defaults to libmoma.so
#    make libmoma.so - build the magpy C++ library
#    make tests - build C++ unit tests
#    make run-tests - build and run C++ unit tests
#    make convergence - build convergence tests
#
#    make clean - cleans up object files
#    make clean-tests - cleans up testing object files
#    make clean-all - cleans all object files
#
CXX=icpc
INC_PATH=include
OBJ_PATH=objects
LIB_PATH=lib

GTEST_DIR=googletest
TESTS=test/test.cpp
TEST_LIB=libgtest.a
GTEST_FLAGS=-isystem $(GTEST_DIR)/include

GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)

ifeq ($(CXX),icpc)
	override CXXFLAGS+=--std=c++11 -W -Wall -pedantic -pthread -O3 -g -fopenmp -simd -qopenmp -xHost -DVERSION=\"$(GIT_VERSION)\" -DUSEMKL -DMKL_ILP64 -I$(MKLROOT)/include -wd488 -wd10145
	LDLIBS=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
else
	override CXXFLAGS+=--std=c++11 -W -Wall -pthread -pedantic -O3 -g -DVERSION=\"$(GIT_VERSION)\"
	LDLIBS=-lopenblas
endif

SOURCES=$(wildcard $(LIB_PATH)/*.cpp)
OBJ_FILES=$(addprefix $(OBJ_PATH)/,$(notdir $(SOURCES:.cpp=.o)))

# Default target builds the magpy library
default: libmoma.so

libmoma.so: $(OBJ_FILES)
	$(CXX) 	-shared -o $@ $^ \
		$(LDFLAGS) $(LDLIBS)

# Build the individual object files
$(OBJ_PATH)/%.o: $(LIB_PATH)/%.cpp
	$(CXX)	$(CXXFLAGS) -c -fPIC \
		-o $@ $<

# Run the unit tests only
run-tests: tests
	cd test && ./tests

tests: test/tests

# The unit tests are run using googletest
test/tests: test/tests.cpp libmoma.so $(GTEST_HEADERS) test/gtest_main.a
	$(CXX) 	$(GTEST_FLAGS) $(CXXFLAGS) $< test/gtest_main.a \
		-o $@ $(LDFLAGS) -L. -lmoma

# Additional test-suit tests
CONVERGENCE_SOURCES=$(wildcard test/convergence/*.cpp)
test/convergence/run: $(CONVERGENCE_SOURCES) $(OBJ_FILES)
	$(CXX) 	$(CXXFLAGS) $(LDFLAGS) test/convergence/all_tasks.cpp \
		$(OBJ_FILES) \
		-o $@ $(LDLIBS)

convergence: test/convergence/run

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

# Clean up
clean:
	rm -f objects/*

clean-tests:
	rm -f test/*.o test/*.a

clean-all: clean clean-tests
