CXX=g++
INC_PATH=include
OBJ_PATH=objects
LIB_PATH=lib
SRC_PATH=src

CPP_FLAGS=--std=c++11 -W -Wall -pedantic -g

SOURCES=$(wildcard $(LIB_PATH)/*.cpp)
OBJ_FILES=$(addprefix $(OBJ_PATH)/,$(notdir $(SOURCES:.cpp=.o)))

default: main

main: src/main.cpp $(OBJ_FILES)
	$(CXX) $(CPP_FLAGS) $< \
	$(OBJ_FILES) \
	-o $@

$(OBJ_PATH)/%.o: $(LIB_PATH)/%.cpp
	$(CXX) 	$(CPP_FLAGS) -c \
	-o $@ $<

clean:
	rm -f objects/*
	rm -f main
