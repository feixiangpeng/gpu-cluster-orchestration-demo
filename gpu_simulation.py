import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

class MatrixMultiplicationSimulator:
    def __init__(self):
        self.sizes = [100, 500, 1000, 2000, 3000]
        self.cpu_times = []
        self.gpu_times = []
        self.speedups = []
    
    def cpu_matrix_multiply(self, size):
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        start_time = time.time()
        C = np.matmul(A, B)
        end_time = time.time()
        
        return end_time - start_time
    
    def gpu_matrix_multiply(self, size):
        transfer_time = size**2 * 8 * 2 / (10 * 1024 * 1024 * 1024)
        compute_factor = 30 * (1 - np.exp(-size/500))
        compute_time = self.cpu_matrix_multiply(size) / compute_factor
        transfer_back = size**2 * 8 / (10 * 1024 * 1024 * 1024)
        
        return transfer_time + compute_time + transfer_back
    
    def run_simulation(self):
        for size in self.sizes:
            print(f"Running simulation for matrix size {size}x{size}...")
            
            cpu_time = self.cpu_matrix_multiply(size)
            self.cpu_times.append(cpu_time)
            print(f"  CPU time: {cpu_time:.4f} seconds")
            
            gpu_time = self.gpu_matrix_multiply(size)
            self.gpu_times.append(gpu_time)
            print(f"  GPU time: {gpu_time:.4f} seconds")
            
            speedup = cpu_time / gpu_time
            self.speedups.append(speedup)
            print(f"  Speedup: {speedup:.2f}x\n")
    
    def visualize_results(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.title("CPU vs GPU Matrix Multiplication Performance")
        plt.plot(self.sizes, self.cpu_times, 'o-', label='CPU')
        plt.plot(self.sizes, self.gpu_times, 'o-', label='GPU (simulated)')
        plt.xlabel("Matrix Size")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.title("GPU Speedup Factor")
        plt.plot(self.sizes, self.speedups, 'o-', color='green')
        plt.xlabel("Matrix Size")
        plt.ylabel("Speedup (CPU time / GPU time)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("gpu_vs_cpu_performance.png")
        plt.close()

def visualize_parallelism():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cpu_data = np.zeros((10, 10))
    cpu_data[0, :] = 1
    ax1.imshow(cpu_data, cmap='Blues')
    ax1.set_title('CPU Processing (Sequential)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    gpu_data = np.ones((10, 10))
    ax2.imshow(gpu_data, cmap='Reds')
    ax2.set_title('GPU Processing (Parallel)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig("cpu_vs_gpu_parallelism.png")
    plt.close()

def animate_gpu_computing():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    cpu_data = np.zeros((10, 10))
    gpu_data = np.zeros((10, 10))
    
    def init():
        ax1.imshow(cpu_data, cmap='Blues', vmin=0, vmax=1)
        ax1.set_title('CPU Processing')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ax2.imshow(gpu_data, cmap='Reds', vmin=0, vmax=1)
        ax2.set_title('GPU Processing')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        return ax1, ax2
    
    def update(frame):
        cpu_row = frame // 10
        cpu_col = frame % 10
        if cpu_row < 10:
            cpu_data[cpu_row, cpu_col] = 1
        
        if frame < 10:
            gpu_data[frame, :] = 1
        
        ax1.imshow(cpu_data, cmap='Blues', vmin=0, vmax=1)
        ax2.imshow(gpu_data, cmap='Reds', vmin=0, vmax=1)
        
        return ax1, ax2
    
    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100)
    plt.tight_layout()
    plt.savefig("gpu_vs_cpu_animation.png")
    plt.close()

def main():
    simulator = MatrixMultiplicationSimulator()
    simulator.run_simulation()
    simulator.visualize_results()
    
    visualize_parallelism()
    animate_gpu_computing()
    
    print("Simulation complete. Output files:")
    print("1. gpu_vs_cpu_performance.png - Performance comparison charts")
    print("2. cpu_vs_gpu_parallelism.png - Visual representation of parallelism")
    print("3. gpu_vs_cpu_animation.png - Animation showing processing differences")

if __name__ == "__main__":
    main()