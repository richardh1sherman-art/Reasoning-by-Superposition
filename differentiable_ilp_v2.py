import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class DeepGeometricILP(nn.Module):
    def __init__(self, num_entities, num_predicates, embedding_dim=8):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 8-Dimensional dense vector embeddings for continuous thought
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(self.device)
        self.predicate_embeddings = nn.Embedding(num_predicates, embedding_dim).to(self.device)
        
        # Soft-clause superposition matrix grid
        self.clause_weights = nn.Parameter(torch.randn(num_predicates, num_predicates, device=self.device))
        
    def forward(self, positive_triples, negative_triples):
        soft_rules = torch.softmax(self.clause_weights, dim=-1)
        loss = 0.0
        
        for sub, rel, obj in positive_triples:
            v_sub = self.entity_embeddings(torch.tensor(sub, device=self.device))
            v_rel = self.predicate_embeddings(torch.tensor(rel, device=self.device))
            v_obj = self.entity_embeddings(torch.tensor(obj, device=self.device))
            loss += torch.sum((v_sub + v_rel - v_obj) ** 2) * torch.mean(soft_rules[rel])
            
        for sub, rel, obj in negative_triples:
            v_sub = self.entity_embeddings(torch.tensor(sub, device=self.device))
            v_rel = self.predicate_embeddings(torch.tensor(rel, device=self.device))
            v_obj = self.entity_embeddings(torch.tensor(obj, device=self.device))
            base_loss = torch.sum((v_sub + v_rel - v_obj) ** 2)
            loss += torch.clamp(4.0 - base_loss, min=0.0) * torch.mean(soft_rules[rel])
            
        return loss

def scale_and_decompile():
    print("🚀 Booting Scale-Up Layer: Injecting Real Boundaries...")
    
    # Text lookup dictionaries for our automatic de-compiler translator
    entity_names = {0: "point_A", 1: "point_B", 2: "line_1", 3: "line_2", 4: "plane_X"}
    pred_names = {0: "parallel", 1: "perpendicular", 2: "orthogonal", 3: "intersects"}
    
    pos_examples = [(0, 0, 2), (1, 0, 3), (2, 2, 4)]
    neg_examples = [(0, 1, 3), (3, 2, 1)]
    
    model = DeepGeometricILP(num_entities=5, num_predicates=4, embedding_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.04)
    
    # Increase epochs to 3000 to give nvidia-smi an actual window to capture the compute spike
    t0 = time.perf_counter()
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = model(pos_examples, neg_examples)
        if loss.item() < 1e-5:
            break
        loss.backward()
        optimizer.step()
        
    speed = time.perf_counter() - t0
    print(f"\nConvergence complete in {speed:.4f}s | Global Energy Floor: {loss.item():.6f}")
    
    # --- AUTOMATED TEXT DE-COMPILER ---
    with torch.no_grad():
        rules = torch.softmax(model.clause_weights, dim=-1).cpu().numpy()
        print("\n🔮 De-Compiling Continuous Probabilities into Pure Text Logic:\n")
        
        for head_idx, row in enumerate(rules):
            body_idx = np.argmax(row)
            confidence = row[body_idx] * 100
            
            head_str = pred_names[head_idx]
            body_str = pred_names[body_idx]
            
            # Print cleanly formatted Prolog Horn clause format syntax
            print(f"  {head_str}(X, Y) :- {body_str}(X, Y).   ({confidence:.1f}% Confidence Match)")

if __name__ == "__main__":
    scale_and_decompile()
