import random
import itertools
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ProblemType(Enum):
    RANDOM_3SAT = "random_3sat"
    PIGEONHOLE = "pigeonhole"
    GRAPH_COLORING = "graph_coloring"
    SCHEDULING = "scheduling"


@dataclass
class Clause:
    literals: List[int]  # Positive for variable, negative for negation
    
    def __str__(self):
        return " ∨ ".join([f"x{abs(lit)}" if lit > 0 else f"¬x{abs(lit)}" for lit in self.literals])


@dataclass
class SATInstance:
    num_variables: int
    clauses: List[Clause]
    problem_type: ProblemType
    metadata: Dict
    
    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.num_variables} {len(self.clauses)}"]
        for clause in self.clauses:
            lines.append(" ".join(map(str, clause.literals)) + " 0")
        return "\n".join(lines)


class SATGenerator:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed) if seed is not None else random
    
    def generate_random_3sat(self, num_variables: int, num_clauses: int) -> SATInstance:
        clauses = []
        for _ in range(num_clauses):
            variables = self._rng.sample(range(1, num_variables + 1), 3)
            literals = [var if self._rng.choice([True, False]) else -var for var in variables]
            clauses.append(Clause(literals))
        
        return SATInstance(
            num_variables=num_variables,
            clauses=clauses,
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={
                "clause_to_variable_ratio": num_clauses / num_variables,
                "difficulty_estimate": "hard" if num_clauses / num_variables > 4.2 else "easy"
            }
        )
    
    def generate_pigeonhole_principle(self, num_pigeons: int, num_holes: int) -> SATInstance:
        if num_pigeons <= num_holes:
            raise ValueError("Pigeonhole principle requires more pigeons than holes")
        
        num_variables = num_pigeons * num_holes
        clauses = []
        
        # Each pigeon must be in at least one hole
        for pigeon in range(num_pigeons):
            literals = [pigeon * num_holes + hole + 1 for hole in range(num_holes)]
            clauses.append(Clause(literals))
        
        # No two pigeons can be in the same hole
        for hole in range(num_holes):
            for p1 in range(num_pigeons):
                for p2 in range(p1 + 1, num_pigeons):
                    var1 = p1 * num_holes + hole + 1
                    var2 = p2 * num_holes + hole + 1
                    clauses.append(Clause([-var1, -var2]))
        
        return SATInstance(
            num_variables=num_variables,
            clauses=clauses,
            problem_type=ProblemType.PIGEONHOLE,
            metadata={
                "num_pigeons": num_pigeons,
                "num_holes": num_holes,
                "unsatisfiable": True,
                "difficulty_estimate": "hard"
            }
        )
    
    def generate_graph_coloring(self, num_vertices: int, edges: List[Tuple[int, int]], num_colors: int) -> SATInstance:
        num_variables = num_vertices * num_colors
        clauses = []
        
        # Each vertex must have at least one color
        for vertex in range(num_vertices):
            literals = [vertex * num_colors + color + 1 for color in range(num_colors)]
            clauses.append(Clause(literals))
        
        # Each vertex can have at most one color
        for vertex in range(num_vertices):
            for c1 in range(num_colors):
                for c2 in range(c1 + 1, num_colors):
                    var1 = vertex * num_colors + c1 + 1
                    var2 = vertex * num_colors + c2 + 1
                    clauses.append(Clause([-var1, -var2]))
        
        # Adjacent vertices cannot have the same color
        for v1, v2 in edges:
            for color in range(num_colors):
                var1 = v1 * num_colors + color + 1
                var2 = v2 * num_colors + color + 1
                clauses.append(Clause([-var1, -var2]))
        
        return SATInstance(
            num_variables=num_variables,
            clauses=clauses,
            problem_type=ProblemType.GRAPH_COLORING,
            metadata={
                "num_vertices": num_vertices,
                "num_edges": len(edges),
                "num_colors": num_colors,
                "chromatic_number_upper_bound": num_colors
            }
        )
    
    def generate_scheduling_problem(self, num_jobs: int, num_time_slots: int, 
                                  conflicts: List[Tuple[int, int]]) -> SATInstance:
        num_variables = num_jobs * num_time_slots
        clauses = []
        
        # Each job must be scheduled in exactly one time slot
        for job in range(num_jobs):
            literals = [job * num_time_slots + slot + 1 for slot in range(num_time_slots)]
            clauses.append(Clause(literals))
            
            for s1 in range(num_time_slots):
                for s2 in range(s1 + 1, num_time_slots):
                    var1 = job * num_time_slots + s1 + 1
                    var2 = job * num_time_slots + s2 + 1
                    clauses.append(Clause([-var1, -var2]))
        
        # Conflicting jobs cannot be scheduled at the same time
        for job1, job2 in conflicts:
            for slot in range(num_time_slots):
                var1 = job1 * num_time_slots + slot + 1
                var2 = job2 * num_time_slots + slot + 1
                clauses.append(Clause([-var1, -var2]))
        
        return SATInstance(
            num_variables=num_variables,
            clauses=clauses,
            problem_type=ProblemType.SCHEDULING,
            metadata={
                "num_jobs": num_jobs,
                "num_time_slots": num_time_slots,
                "num_conflicts": len(conflicts)
            }
        )
    
    def generate_diverse_instances(self, count: int = 100) -> List[SATInstance]:
        instances = []
        
        for i in range(count):
            problem_type = self._rng.choice(list(ProblemType))
            
            if problem_type == ProblemType.RANDOM_3SAT:
                num_vars = self._rng.randint(10, 50)
                num_clauses = self._rng.randint(int(2 * num_vars), int(6 * num_vars))
                instance = self.generate_random_3sat(num_vars, num_clauses)
            
            elif problem_type == ProblemType.PIGEONHOLE:
                num_holes = self._rng.randint(3, 8)
                num_pigeons = num_holes + self._rng.randint(1, 3)
                instance = self.generate_pigeonhole_principle(num_pigeons, num_holes)
            
            elif problem_type == ProblemType.GRAPH_COLORING:
                num_vertices = self._rng.randint(5, 15)
                edge_prob = self._rng.uniform(0.2, 0.7)
                edges = [(i, j) for i in range(num_vertices) 
                        for j in range(i + 1, num_vertices)
                        if self._rng.random() < edge_prob]
                num_colors = self._rng.randint(2, num_vertices // 2)
                instance = self.generate_graph_coloring(num_vertices, edges, num_colors)
            
            elif problem_type == ProblemType.SCHEDULING:
                num_jobs = self._rng.randint(5, 20)
                num_slots = self._rng.randint(2, num_jobs // 2)
                conflict_prob = self._rng.uniform(0.1, 0.4)
                conflicts = [(i, j) for i in range(num_jobs)
                           for j in range(i + 1, num_jobs)
                           if self._rng.random() < conflict_prob]
                instance = self.generate_scheduling_problem(num_jobs, num_slots, conflicts)
            
            instances.append(instance)
        
        return instances