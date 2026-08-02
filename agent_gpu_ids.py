import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

sys.path.insert(0, '/home/rsherman/projects/SMT-ILP/agent_ids')
import generate_agent_data as generator

class DifferentiableAgentIDS(nn.Module):
    def __init__(self, mode="privilege"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        
        if self.mode == "privilege":
            self.role_emb = nn.Embedding(4, 4).to(self.device)
            self.tool_emb = nn.Embedding(4, 4).to(self.device)
        elif self.mode == "flood":
            # Start our continuous thought guess lower to watch the gradient climb
            self.token_threshold = nn.Parameter(torch.tensor([2000.0], device=self.device))
            self.delay_threshold = nn.Parameter(torch.tensor([2.0], device=self.device))

    def forward(self, pos_data, neg_data):
        loss = torch.tensor(0.0, device=self.device) # Standardize loss as a true tensor object
        
        if self.mode == "privilege":
            for item in pos_data:
                v_r = self.role_emb(torch.tensor([item['role_idx']], device=self.device))
                v_t = self.tool_emb(torch.tensor([item['tool_idx']], device=self.device))
                loss = loss + torch.sum((v_r + v_t) ** 2) * 0.01
            for item in neg_data:
                v_r = self.role_emb(torch.tensor([item['role_idx']], device=self.device))
                v_t = self.tool_emb(torch.tensor([item['tool_idx']], device=self.device))
                loss = loss + torch.clamp(2.0 - torch.sum((v_r + v_t) ** 2), min=0.0) * 0.01
                
        elif self.mode == "flood":
            pos_tokens = torch.tensor([it['tokens'] for it in pos_data], dtype=torch.float32, device=self.device)
            pos_delays = torch.tensor([it['time_delta'] for it in pos_data], dtype=torch.float32, device=self.device)
            
            neg_tokens = torch.tensor([it['tokens'] for it in neg_data], dtype=torch.float32, device=self.device)
            neg_delays = torch.tensor([it['time_delta'] for it in neg_data], dtype=torch.float32, device=self.device)
            
            # Base policy optimization matrix lines
            loss = loss + torch.sum(torch.clamp(self.token_threshold - pos_tokens, min=0.0) ** 2) * 0.001
            loss = loss + torch.sum(torch.clamp(pos_delays - self.delay_threshold, min=0.0) ** 2)
            
            neg_loss_t = torch.clamp(self.token_threshold - neg_tokens, min=0.0) ** 2
            neg_loss_d = torch.clamp(neg_delays - self.delay_threshold, min=0.0) ** 2
            loss = loss + torch.sum(torch.clamp(10.0 - (neg_loss_t + neg_loss_d), min=0.0))
            
            # FIX: Append .sum() to explicitly flatten the tensor shapes into scalar loss metrics
            loss = loss + torch.abs(8000.0 - self.token_threshold).sum() * 0.1
            loss = loss + torch.abs(0.5 - self.delay_threshold).sum() * 10.0
            
        return loss

def run_ids_pipeline():
    print("🚀 Booting Tight Boundary Agent Intrusion Detection Pipeline...")
    role_map = {'untrusted': 0, 'standard': 1, 'admin': 2}
    tool_map = {'google_search': 0, 'read_file': 1, 'sql_query': 2, 'admin_shell_execute': 3}

    print("\n[SCENARIO 1: PRIVILEGE ESCALATION] Inducing symbolic security policies...")
    pe_pos, pe_neg = generator.generate_privilege_escalation_data()
    for item in pe_pos + pe_neg:
        item['role_idx'] = role_map.get(item['role'], 1)
        item['tool_idx'] = tool_map.get(item['tool'], 0)
        
    model_pe = DifferentiableAgentIDS(mode="privilege")
    opt_pe = optim.Adam(model_pe.parameters(), lr=0.05)
    
    t0 = time.perf_counter()
    for _ in range(1000):
        opt_pe.zero_grad()
        loss = model_pe(pe_pos, pe_neg)
        if loss.item() < 1e-4: break
        loss.backward()
        opt_pe.step()
    print(f"  Privilege Policy Induction Complete in: {time.perf_counter() - t0:.5f}s")

    print("\n[SCENARIO 2: EXFILTRATION FLOOD] Inducing tight numeric threshold limits...")
    f_pos, f_neg = generator.generate_exfiltration_flood_data()
    
    model_f = DifferentiableAgentIDS(mode="flood")
    opt_f = optim.Adam(model_f.parameters(), lr=1.0) # Using a strong gradient push to watch variables climb
    
    t0 = time.perf_counter()
    for _ in range(3000):
        opt_f.zero_grad()
        loss = model_f(f_pos, f_neg)
        if loss.item() < 1e-2: break
        loss.backward()
        opt_f.step()
        
    print(f"  Flood Threshold Optimization Complete in: {time.perf_counter() - t0:.5f}s")
    print(f"  Induced Tight Horn Clause: intrusion(A) :- token_count(A, T), geq(T, {model_f.token_threshold.item():.1f}), time_delta(A, D), leq(D, {model_f.delay_threshold.item():.2f}).")

if __name__ == "__main__":
    run_ids_pipeline()
