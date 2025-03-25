import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class GPUNode:
    def __init__(self, node_id, num_gpus=4, gpu_mem=16):
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.gpu_mem = gpu_mem
        self.available_gpus = num_gpus
        self.jobs = []
        self.utilization_history = []
        self.timestamp_history = []
    
    def can_accommodate(self, job):
        return job.gpu_count <= self.available_gpus
    
    def allocate(self, job):
        if self.can_accommodate(job):
            self.available_gpus -= job.gpu_count
            self.jobs.append(job)
            job.node_id = self.node_id
            job.status = "RUNNING"
            job.start_time = datetime.now()
            return True
        return False
    
    def update(self):
        current_time = datetime.now()
        completed_jobs = []
        
        for job in self.jobs:
            elapsed = (current_time - job.start_time).total_seconds()
            if elapsed >= job.duration:
                job.status = "COMPLETED"
                self.available_gpus += job.gpu_count
                completed_jobs.append(job)
        
        for job in completed_jobs:
            self.jobs.remove(job)
        
        utilization = (self.num_gpus - self.available_gpus) / self.num_gpus
        self.utilization_history.append(utilization)
        self.timestamp_history.append(current_time)
        
        return completed_jobs

class Job:
    def __init__(self, job_id, gpu_count, duration, priority=1, mem_req=4):
        self.job_id = job_id
        self.gpu_count = gpu_count
        self.duration = duration
        self.priority = priority
        self.mem_req = mem_req
        self.status = "QUEUED"
        self.node_id = None
        self.submit_time = datetime.now()
        self.start_time = None
        self.completion_time = None
    
    def __str__(self):
        return f"Job {self.job_id}: {self.gpu_count} GPUs, {self.duration}s, Priority {self.priority}, Status: {self.status}"

class ClusterScheduler:
    def __init__(self, num_nodes=4, gpus_per_node=4):
        self.nodes = [GPUNode(i, gpus_per_node) for i in range(num_nodes)]
        self.job_queue = []
        self.completed_jobs = []
        self.next_job_id = 1
        self.overall_utilization = []
        self.time_points = []
    
    def submit_job(self, gpu_count, duration_seconds, priority=1, mem_req=4):
        job = Job(self.next_job_id, gpu_count, duration_seconds, priority, mem_req)
        self.next_job_id += 1
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: (-x.priority, x.submit_time))
        return job
    
    def schedule_jobs(self):
        if not self.job_queue:
            return 0
        
        jobs_scheduled = 0
        remaining_jobs = []
        
        for job in self.job_queue:
            scheduled = False
            
            for node in self.nodes:
                if node.allocate(job):
                    scheduled = True
                    jobs_scheduled += 1
                    break
            
            if not scheduled:
                remaining_jobs.append(job)
        
        self.job_queue = remaining_jobs
        return jobs_scheduled
    
    def update_cluster(self):
        newly_completed = []
        
        for node in self.nodes:
            completed = node.update()
            newly_completed.extend(completed)
        
        for job in newly_completed:
            job.completion_time = datetime.now()
            self.completed_jobs.append(job)
        
        total_gpus = sum(node.num_gpus for node in self.nodes)
        used_gpus = sum(node.num_gpus - node.available_gpus for node in self.nodes)
        utilization = used_gpus / total_gpus if total_gpus > 0 else 0
        
        self.overall_utilization.append(utilization)
        self.time_points.append(datetime.now())
        
        return newly_completed
    
    def get_status(self):
        queued = len(self.job_queue)
        running = sum(len(node.jobs) for node in self.nodes)
        completed = len(self.completed_jobs)
        
        return {
            "queued": queued,
            "running": running,
            "completed": completed,
            "total": queued + running + completed
        }
    
    def visualize_utilization(self):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.title("Overall Cluster GPU Utilization")
        plt.plot([t.timestamp() for t in self.time_points], 
                 self.overall_utilization, 'b-', label='Overall')
        plt.ylim(0, 1.1)
        plt.ylabel("Utilization")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.title("Per-Node GPU Utilization")
        for i, node in enumerate(self.nodes):
            plt.plot([t.timestamp() for t in node.timestamp_history], 
                     node.utilization_history, label=f'Node {node.node_id}')
        plt.ylim(0, 1.1)
        plt.ylabel("Utilization")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("cluster_utilization.png")
        plt.close()

def print_status(scheduler):
    status = scheduler.get_status()
    print(f"\nCluster Status: {status['queued']} queued, {status['running']} running, {status['completed']} completed")
    
    print("\nRunning Jobs:")
    for node in scheduler.nodes:
        if node.jobs:
            print(f"Node {node.node_id}: {node.num_gpus - node.available_gpus}/{node.num_gpus} GPUs used")
            for job in node.jobs:
                print(f"  {job}")
    
    print("\nQueued Jobs:")
    for job in scheduler.job_queue:
        print(f"  {job}")

def main():
    scheduler = ClusterScheduler(num_nodes=4, gpus_per_node=4)
    
    scheduler.submit_job(gpu_count=2, duration_seconds=5, priority=2)
    scheduler.submit_job(gpu_count=1, duration_seconds=8, priority=1)
    scheduler.submit_job(gpu_count=4, duration_seconds=3, priority=3)
    scheduler.submit_job(gpu_count=3, duration_seconds=7, priority=2)
    
    simulation_time = 15
    start_time = time.time()
    
    while time.time() - start_time < simulation_time:
        scheduled = scheduler.schedule_jobs()
        if scheduled:
            print(f"Scheduled {scheduled} new jobs")
        
        completed = scheduler.update_cluster()
        if completed:
            print(f"Completed {len(completed)} jobs")
        
        print_status(scheduler)
        
        if time.time() - start_time > 5 and time.time() - start_time < 6:
            print("\nSubmitting a new batch of jobs...")
            scheduler.submit_job(gpu_count=2, duration_seconds=3, priority=1)
            scheduler.submit_job(gpu_count=1, duration_seconds=4, priority=3)
        
        time.sleep(1)
    
    scheduler.visualize_utilization()
    print("\nSimulation completed. Utilization graph saved as 'cluster_utilization.png'")

if __name__ == "__main__":
    main()