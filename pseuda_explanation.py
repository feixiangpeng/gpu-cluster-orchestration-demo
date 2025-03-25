import numpy as np
import matplotlib.pyplot as plt
import time

def explain_cuda_concepts():
    print("=" * 80)
    print("CUDA PROGRAMMING CONCEPTS EXPLANATION".center(80))
    print("=" * 80)
    
    print("\n1. CUDA EXECUTION MODEL\n")
    print("In CUDA, computation is organized in:")
    print("  - GRID: The entire computation")
    print("  - BLOCKS: Groups of threads that execute together")
    print("  - THREADS: Individual execution units")
    
    print("\nPseudo-CUDA kernel definition:")
    print("""
    __global__ void matrixMultiply(float* A, float* B, float* C, int width) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < width && col < width) {
            float sum = 0.0f;
            for (int i = 0; i < width; i++) {
                sum += A[row * width + i] * B[i * width + col];
            }
            C[row * width + col] = sum;
        }
    }
    """)
    
    print("\nKernel launch configuration:")
    print("""
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    """)
    
    print("\n2. MEMORY HIERARCHY\n")
    print("CUDA devices have multiple memory types:")
    print("  - Global Memory: Accessible by all threads, highest latency")
    print("  - Shared Memory: Shared within a block, much faster than global")
    print("  - Registers: Per-thread, fastest access")
    print("  - Constant Memory: Read-only, cached, good for lookup tables")
    print("  - Texture Memory: Optimized for 2D spatial locality")
    
    print("\nShared memory example:")
    print("""
    __global__ void sharedMemMatrixMult(float* A, float* B, float* C, int width) {
        __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
        __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
        
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        float sum = 0.0f;
        
        for (int t = 0; t < width/TILE_SIZE; t++) {
            sharedA[threadIdx.y][threadIdx.x] = A[row*width + t*TILE_SIZE + threadIdx.x];
            sharedB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*width + col];
            
            __syncthreads();
            
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        C[row*width + col] = sum;
    }
    """)
    
    print("\n3. THREAD SYNCHRONIZATION\n")
    print("CUDA provides synchronization primitives:")
    print("  - __syncthreads(): Synchronizes all threads in a block")
    print("  - Atomic operations: For thread-safe updates to shared values")
    
    print("\nAtomic operation example:")
    print("""
    __global__ void histogram(int* data, int* hist, int N) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < N) {
            int value = data[tid];
            atomicAdd(&hist[value], 1);
        }
    }
    """)
    
    print("\n4. MEMORY TRANSFERS\n")
    print("Data must be transferred between host (CPU) and device (GPU):")
    print("""
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    """)

def visualize_cuda_execution_model():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    grid_rect = plt.Rectangle((0, 0), 8, 8, fc='none', ec='black', lw=2)
    ax.add_patch(grid_rect)
    ax.text(4, 8.5, "GRID", horizontalalignment='center', fontsize=14)
    
    for i in range(4):
        for j in range(4):
            block_rect = plt.Rectangle((i*2, j*2), 2, 2, fc='none', ec='blue', lw=1.5)
            ax.add_patch(block_rect)
            ax.text(i*2+1, j*2+1, f"Block\n({i},{j})", horizontalalignment='center', 
                   verticalalignment='center', fontsize=10, color='blue')
    
    zoom_ax = fig.add_axes([0.65, 0.15, 0.3, 0.3])
    zoom_ax.set_xlim(0, 4)
    zoom_ax.set_ylim(0, 4)
    zoom_rect = plt.Rectangle((0, 0), 4, 4, fc='none', ec='blue', lw=1.5)
    zoom_ax.add_patch(zoom_rect)
    zoom_ax.text(2, 4.5, "Block (0,0) - Threads", horizontalalignment='center', fontsize=10)
    
    for i in range(4):
        for j in range(4):
            thread_rect = plt.Rectangle((i, j), 1, 1, fc='none', ec='red', lw=1)
            zoom_ax.add_patch(thread_rect)
            zoom_ax.text(i+0.5, j+0.5, f"({i},{j})", horizontalalignment='center', 
                       verticalalignment='center', fontsize=8, color='red')
    
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    zoom_ax.axis('off')
    
    plt.savefig("cuda_execution_model.png")
    plt.close()

def visualize_memory_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hierarchy = [
        ("Registers\n(Per Thread)", "darkgreen", 1),
        ("Shared Memory\n(Per Block)", "green", 2),
        ("L1/L2 Cache", "lightgreen", 3),
        ("Global Memory\n(Device VRAM)", "lightblue", 4)
    ]
    
    for i, (name, color, size) in enumerate(hierarchy):
        left = i * 0.5
        rect = plt.Rectangle((left, 0), 10 - i, size, fc=color, ec='black', lw=1)
        ax.add_patch(rect)
        
        ax.text(5, 0.5 + sum(item[2] for item in hierarchy[:i]), name, 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=12, fontweight='bold')
        
        if i < len(hierarchy) - 1:
            ax.text(9, 0.5 + sum(item[2] for item in hierarchy[:i]), 
                    f"{10**(len(hierarchy)-i-1)}x faster", 
                    horizontalalignment='right', verticalalignment='center', 
                    fontsize=10, color='darkred')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    plt.title("CUDA Memory Hierarchy", fontsize=16)
    
    plt.savefig("cuda_memory_hierarchy.png")
    plt.close()

def main():
    explain_cuda_concepts()
    visualize_cuda_execution_model()
    visualize_memory_hierarchy()
    
    print("\nExplanation complete. Visual diagrams saved as:")
    print("1. cuda_execution_model.png - Visual representation of CUDA's grid/block/thread model")
    print("2. cuda_memory_hierarchy.png - CUDA memory hierarchy diagram")

if __name__ == "__main__":
    main()