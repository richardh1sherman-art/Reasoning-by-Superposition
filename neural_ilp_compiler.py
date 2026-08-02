import torch
import torch.optim as optim
import numpy as np
import re

class ContinuousThoughtCompiler:
    """
    Accepts Prolog-style Horn Clauses from PyGol.
    Compiles text constraints into spatial vector superpositions on the GPU.
    """
    def __init__(self, embedding_dim=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = embedding_dim
        self.entities = {}
        self.relations = {}
        
    def _get_embedding(self, name, dictionary, requires_grad=True):
        if name not in dictionary:
            vec = torch.randn(self.dim, device=self.device, requires_grad=requires_grad)
            dictionary[name] = vec
        return dictionary[name]

    def evaluate_horn_clause(self, clause_str: str, steps=500) -> bool:
        """
        Compiles a discrete rule: Head :- Body1, Body2...
        and optimizes the spatial map using gradient descent.
        """
        # ADD THIS EXPLICIT PRINT STATEMENT TO TRACK INTERCEPTED PROCESS DATA:
        print(f"  🧠 [CONTINUOUS THOUGHT ACTIVATED] Compiling Superposition Map for: {clause_str}")

        clause_str = clause_str.replace(" ", "").replace(".", "")
        if ":-" not in clause_str:
            return True
            
        head, body = clause_str.split(":-")
        body_literals = body.split(",")
        
        active_params = []
        
        # FIX: Split the malformed syntax assignment loop into clean separate lines
        head_match = re.match(r'(\w+)\((.*)\)', head)
        if not head_match: 
            return False
            
        head_rel = head_match.group(1)
        head_args_list = head_match.group(2).split(",")
        
        v_head_rel = self._get_embedding(head_rel, self.relations)
        active_params.append(v_head_rel)
        
        # Setup the Optimization Loop
        optimizer = optim.Adam(active_params, lr=0.1)
        
        for _ in range(steps):
            optimizer.zero_grad()
            total_energy = 0.0
            
            for literal in body_literals:
                lit_match = re.match(r'(\w+)\((.*)\)', literal)
                if not lit_match: 
                    continue
                rel_name, args = lit_match.group(1), lit_match.group(2).split(",")
                
                if len(args) == 2:
                    v_sub = self._get_embedding(args[0], self.entities)
                    v_rel = self._get_embedding(rel_name, self.relations)
                    v_obj = self._get_embedding(args[1], self.entities)
                    
                    if v_sub.requires_grad: active_params.append(v_sub)
                    if v_rel.requires_grad: active_params.append(v_rel)
                    if v_obj.requires_grad: active_params.append(v_obj)
                    
                    total_energy += torch.sum((v_sub + v_rel - v_obj) ** 2)
                    
            if total_energy.item() < 1e-4:
                return True
                
            total_energy.backward(retain_graph=True)
            optimizer.step()
            
        return total_energy.item() < 1e-3

compiler_engine = ContinuousThoughtCompiler()
