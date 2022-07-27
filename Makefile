CUDA_ROOT_DIR = /usr/local/cuda
CUDA_LIB_DIR = $(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR = $(CUDA_ROOT_DIR)/include

SRC_DIR = src
OBJ_DIR = src
INC_DIR = include

EXE = kmeans_seg_npp
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/image_io.o $(OBJ_DIR)/image_copy.o \
       $(OBJ_DIR)/image_conversion.o $(OBJ_DIR)/image_kmeans.o  $(OBJ_DIR)/Exceptions.o

LINK_FLAGS = -L$(CUDA_LIB_DIR)
LINK_LIBS = -lcudart -lnppc -lnppidei -lnppist -lnppisu -lnppial -lnppitc -lnppicc -lculibos -lfreeimage

CC = g++
CC_FLAGS = -I$(INC_DIR) -I$(CUDA_INC_DIR)

NVCC = nvcc
NVCC_FLAGS = -I$(INC_DIR) -I$(CUDA_INC_DIR)

.PHONY : clean

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(LINK_FLAGS) $(OBJS) -o $@ $(LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	$(RM) src/*.o $(EXE)
