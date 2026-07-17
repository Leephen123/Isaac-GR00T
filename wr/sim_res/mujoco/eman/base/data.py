import threading
import queue
import csv
import time
from typing import List
import matplotlib.pyplot as plt

class DataSaverThread(threading.Thread):
    
    def __init__(self, data_file: str):
        super().__init__()
        self.data_file = data_file
        self.data_queue = queue.Queue()
        self.running = True
        self.daemon = True
        
        with open(self.data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time', 'joint_idx', 'action', 'dof_pos', 'dof_vel','dof_torque', 'root_rot', 'root_ang_vel']
            writer.writerow(header)
    
    def add_data(self, data_row: List):
        self.data_queue.put(data_row)
    
    def run(self):
        while self.running:
            try:
                data_batch = []
                while len(data_batch) < 10:  
                    try:
                        data_row = self.data_queue.get(timeout=0.1)
                        data_batch.append(data_row)
                    except queue.Empty:
                        break
                
                if data_batch:
                    with open(self.data_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data_batch)
                
            except Exception as e:
                print(f"wrong: {e}")
    
    def stop(self):
        self.running = False
        while not self.data_queue.empty():
            try:
                data_row = self.data_queue.get_nowait()
                with open(self.data_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
            except queue.Empty:
                break

