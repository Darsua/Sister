NVCC = nvcc
NVCCFLAGS = -Iinclude -Xcompiler "-Wall -Wextra -fopenmp -lm -lraylib"
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJS := $(OBJS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/mandelbrot

build: $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET) -lm

rebuild: clean build

.PHONY: clean build rebuild
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
