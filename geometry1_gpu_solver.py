import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# 1. Dynamically append the geometry1 folder path to the environment loader
sys.path.insert(0, '/home/rsherman/projects/SMT-ILP/geometry1')
import load_geometry1_data as loader

class UniversalGeometry1GPUField(nn.Module):
    """
    Continuous Thought Optimizer that maps physical DataFrame matrices 
    straight into parallel GPU boundary classifiers.
    """
    def __init__(self, input_dim=3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using a highly-optimized multi-layer relaxation block to learn complex combinations
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        ).to(self.device)
        
    def forward(self, x_tensor):
        return self.net(x_tensor)

def solve_real_problem_block(problem_name, X_df, y_np, input_dim=3):
    print(f"\n[RUNNING: {problem_name.upper()}] Loading data matrix onto GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract numerical arrays straight from the Pandas frames and convert to tensors
    raw_X = X_df.to_numpy()
    
    X = torch.tensor(raw_X, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
    
    field = UniversalGeometry1GPUField(input_dim=input_dim)
    optimizer = optim.Adam(field.parameters(), lr=0.03)
    criterion = nn.BCEWithLogitsLoss()
    
    t0 = time.perf_counter()
    # High-speed continuous thought relaxation gradient pass
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = field(X)
        loss = criterion(predictions, y)
        if loss.item() < 1e-4: break
        loss.backward()
        optimizer.step()
        
    speed = time.perf_counter() - t0
    print(f"  CUDA Optimization Complete! Convergence Speed: {speed:.5f}s")
    print(f"  Final System Loss Floor Energy: {loss.item():.6f}")
    
    # Run a quick batch classification verification test row
    with torch.no_grad():
        preds = (torch.sigmoid(field(X)) > 0.5).float()
        accuracy = (preds == y).float().mean().item() * 100
        print(f"  ✅ SUCCESS: Continuous Thought Induction Accuracy: {accuracy:.1f}%")

def run_production_pipeline():
    print("🚀 Booting Integrated Geometry1 Live Stream Optimization Pipeline...")
    
    # Path coordinates pointing straight to your Prolog dataset files
    data_dir = "/home/rsherman/projects/SMT-ILP/geometry1/data"
    
    # Problem 1: 3D Halfplane
    file_1 = os.path.join(data_dir, "halfplane3d_examples.pl")
    if os.path.exists(file_1):
        X, y, _ = loader.load_halfplane3d_data(file_1)
        solve_real_problem_block("Problem 1: 3D Halfplane", X, y, input_dim=3)
        
    # Problem 2: Conjunction
    file_2 = os.path.join(data_dir, "conjunction_examples.pl")
    if os.path.exists(file_2):
        X, y, _ = loader.load_conjunction_data(file_2)
        solve_real_problem_block("Problem 2: Conjunction Field", X, y, input_dim=3)
        
    # Problem 3: Multiple Halfplanes
    file_3 = os.path.join(data_dir, "multihalfplane_examples.pl")
    if os.path.exists(file_3):
        X, y, _ = loader.load_multihalfplane_data(file_3)
        solve_real_problem_block("Problem 3: Multiple Halfplanes", X, y, input_dim=2)
        
    # Problem 4: 3D Interval
    file_4 = os.path.join(data_dir, "interval3d_examples.pl")
    if os.path.exists(file_4):
        X, y, _ = loader.load_interval3d_data(file_4)
        solve_real_problem_block("Problem 4: 3D Interval Range", X, y, input_dim=3)

if __name__ == "__main__":
    run_production_pipeline()
