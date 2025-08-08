import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path
import logging
import pickle

from src.core.sat_generator import SATGenerator, ProblemType, SATInstance
from src.core.sat_solver import DPLLSolver, SolverTrace
from src.formatting.trace_formatter import TraceFormatter


@dataclass
class DatasetConfig:
    num_instances: int = 1000
    max_variables_range: Tuple[int, int] = (5, 10)
    problem_type_distribution: Dict[ProblemType, float] = None
    solver_types: List[str] = None
    max_solve_time_seconds: int = 300
    include_unsatisfiable: bool = True
    output_directory: str = "sat_dataset"
    num_processes: int = None
    random_seed: int = 42
    
    def __post_init__(self):
        if self.problem_type_distribution is None:
            self.problem_type_distribution = {
                ProblemType.RANDOM_3SAT: 0.4,
                ProblemType.PIGEONHOLE: 0.2,
                ProblemType.GRAPH_COLORING: 0.2,
                ProblemType.SCHEDULING: 0.2
            }
        
        if self.solver_types is None:
            self.solver_types = ["DPLL"]  # Only DPLL is supported
        
        if self.num_processes is None:
            self.num_processes = 1


class DatasetGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.formatter = TraceFormatter()
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get logger (assumes logging is already configured)
        self.logger = logging.getLogger(__name__)
    
    def generate_single_instance_data(self, instance_id: int, seed: int) -> Optional[Dict[str, Any]]:
        """Generate a single training instance with solver trace"""
        try:
            # Set random seed for reproducibility
            random.seed(seed)
            generator = SATGenerator(seed)
            
            # Generate SAT instance
            problem_type = random.choices(
                list(self.config.problem_type_distribution.keys()),
                weights=list(self.config.problem_type_distribution.values())
            )[0]
            
            min_vars, max_vars = self.config.max_variables_range
            num_vars = random.randint(min_vars, max_vars)
            
            instance = self._generate_instance_by_type(generator, problem_type, num_vars)
            
            if instance is None:
                return None
            
            # Solve with random solver
            solver_type = random.choice(self.config.solver_types)
            trace = self._solve_with_timeout(instance, solver_type, self.config.max_solve_time_seconds)
            
            if trace is None:
                return None
            
            # Skip unsatisfiable instances if not desired
            if not self.config.include_unsatisfiable and trace.final_result is False:
                return None
            
            # Format for training
            training_data = self.formatter.format_trace_for_training(trace)
            training_data["instance_id"] = instance_id
            training_data["solver_type"] = solver_type
            training_data["generation_seed"] = seed
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error generating instance {instance_id}: {str(e)}")
            return None
    
    def _generate_instance_by_type(self, generator: SATGenerator, 
                                 problem_type: ProblemType, num_vars: int) -> Optional[SATInstance]:
        """Generate a SAT instance of the specified type"""
        try:
            if problem_type == ProblemType.RANDOM_3SAT:
                ratio = random.uniform(2.0, 6.0)  # Clause-to-variable ratio
                num_clauses = int(num_vars * ratio)
                return generator.generate_random_3sat(num_vars, num_clauses)
            
            elif problem_type == ProblemType.PIGEONHOLE:
                num_holes = random.randint(3, min(8, num_vars // 2))
                num_pigeons = num_holes + random.randint(1, 3)
                return generator.generate_pigeonhole_principle(num_pigeons, num_holes)
            
            elif problem_type == ProblemType.GRAPH_COLORING:
                num_vertices = random.randint(5, min(15, num_vars))
                edge_prob = random.uniform(0.2, 0.7)
                edges = [(i, j) for i in range(num_vertices) 
                        for j in range(i + 1, num_vertices)
                        if random.random() < edge_prob]
                num_colors = random.randint(2, max(2, num_vertices // 2))
                return generator.generate_graph_coloring(num_vertices, edges, num_colors)
            
            elif problem_type == ProblemType.SCHEDULING:
                num_jobs = random.randint(5, min(20, num_vars))
                num_slots = random.randint(2, max(2, num_jobs // 2))
                conflict_prob = random.uniform(0.1, 0.4)
                conflicts = [(i, j) for i in range(num_jobs)
                           for j in range(i + 1, num_jobs)
                           if random.random() < conflict_prob]
                return generator.generate_scheduling_problem(num_jobs, num_slots, conflicts)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate {problem_type.value} instance: {str(e)}")
            return None
        
        return None
    
    def _solve_with_timeout(self, instance: SATInstance, solver_type: str, 
                          timeout_seconds: int) -> Optional[SolverTrace]:
        """Solve instance with timeout"""
        try:
            start_time = time.time()
            
            if solver_type == "DPLL":
                solver = DPLLSolver(instance)
            else:
                raise ValueError(f"Unknown solver type: {solver_type} (only DPLL is supported)")
            
            # Simple timeout mechanism (for more complex cases, you might want to use signals)
            result = solver.solve()
            
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                self.logger.warning(f"Solver timeout after {elapsed:.2f} seconds")
                return None
            
            trace = solver.get_trace()
            return trace
            
        except Exception as e:
            self.logger.error(f"Error solving instance: {str(e)}")
            return None
    
    def generate_dataset_sequential(self) -> Dict[str, Any]:
        """Generate dataset sequentially (simpler, no multiprocessing issues)"""
        self.logger.info(f"Starting sequential dataset generation")
        self.logger.info(f"Target instances: {self.config.num_instances}")
        
        start_time = time.time()
        successful_instances = []
        failed_count = 0
        
        base_seed = self.config.random_seed
        
        for i in range(self.config.num_instances):
            seed = base_seed + i
            result = self.generate_single_instance_data(i, seed)
            
            if result is not None:
                successful_instances.append(result)
                if len(successful_instances) % 10 == 0:
                    self.logger.info(f"Generated {len(successful_instances)} instances...")
            else:
                failed_count += 1
        
        elapsed_time = time.time() - start_time
        
        # Same statistics compilation as parallel version...
        dataset_info = {
            "generation_config": {
                "num_instances_requested": self.config.num_instances,
                "num_instances_generated": len(successful_instances),
                "num_failed": failed_count,
                "generation_time_seconds": elapsed_time,
                "processes_used": 1,  # Sequential
                "random_seed": self.config.random_seed
            },
            "problem_type_stats": {},
            "satisfiability_stats": {"satisfiable": 0, "unsatisfiable": 0, "unknown": 0},
            "solver_stats": {},
            "complexity_stats": {
                "min_variables": float('inf'),
                "max_variables": 0,
                "min_clauses": float('inf'),
                "max_clauses": 0,
                "avg_steps": 0
            }
        }
        
        # Calculate statistics
        total_steps = 0
        for instance in successful_instances:
            problem_type = instance["problem"]["type"]
            dataset_info["problem_type_stats"][problem_type] = \
                dataset_info["problem_type_stats"].get(problem_type, 0) + 1
            
            sat_result = instance["solution"]["satisfiable"]
            if sat_result is True:
                dataset_info["satisfiability_stats"]["satisfiable"] += 1
            elif sat_result is False:
                dataset_info["satisfiability_stats"]["unsatisfiable"] += 1
            else:
                dataset_info["satisfiability_stats"]["unknown"] += 1
            
            solver_type = instance["solver_type"]
            dataset_info["solver_stats"][solver_type] = \
                dataset_info["solver_stats"].get(solver_type, 0) + 1
            
            num_vars = instance["problem"]["num_variables"]
            num_clauses = instance["problem"]["num_clauses"]
            steps = instance["solution"]["steps_taken"]
            
            dataset_info["complexity_stats"]["min_variables"] = min(
                dataset_info["complexity_stats"]["min_variables"], num_vars)
            dataset_info["complexity_stats"]["max_variables"] = max(
                dataset_info["complexity_stats"]["max_variables"], num_vars)
            dataset_info["complexity_stats"]["min_clauses"] = min(
                dataset_info["complexity_stats"]["min_clauses"], num_clauses)
            dataset_info["complexity_stats"]["max_clauses"] = max(
                dataset_info["complexity_stats"]["max_clauses"], num_clauses)
            
            total_steps += steps
        
        if successful_instances:
            dataset_info["complexity_stats"]["avg_steps"] = total_steps / len(successful_instances)
        
        self.logger.info(f"Sequential dataset generation completed!")
        self.logger.info(f"Generated {len(successful_instances)} successful instances")
        self.logger.info(f"Failed instances: {failed_count}")
        self.logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        
        return {
            "instances": successful_instances,
            "dataset_info": dataset_info
        }
    
    
    def save_dataset(self, dataset: Dict[str, Any], format: str = "json"):
        """Save dataset to disk in specified format"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create run-specific subdirectory
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        if format == "json":
            output_file = run_dir / f"sat_dataset_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
        
        elif format == "jsonl":
            output_file = run_dir / f"sat_dataset_{timestamp}.jsonl"
            with open(output_file, 'w') as f:
                for instance in dataset["instances"]:
                    f.write(json.dumps(instance) + '\n')
        
        elif format == "pickle":
            output_file = run_dir / f"sat_dataset_{timestamp}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(dataset, f)
        
        # Always save metadata
        metadata_file = run_dir / f"dataset_info_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset["dataset_info"], f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_file}")
        self.logger.info(f"Metadata saved to {metadata_file}")
        
        return output_file
    
    def generate_and_save(self, save_formats: List[str] = ["json", "jsonl"]) -> List[Path]:
        """Complete dataset generation and saving pipeline"""
        dataset = self.generate_dataset_sequential()
        
        saved_files = []
        for format in save_formats:
            file_path = self.save_dataset(dataset, format)
            saved_files.append(file_path)
        
        return saved_files



if __name__ == "__main__":
    # Example usage
    config = DatasetConfig(
        num_instances=1000,
        max_variables_range=(10, 30),
        problem_type_distribution={
            ProblemType.RANDOM_3SAT: 0.5,
            ProblemType.PIGEONHOLE: 0.2,
            ProblemType.GRAPH_COLORING: 0.15,
            ProblemType.SCHEDULING: 0.15
        },
        solver_types=["DPLL"],
        max_solve_time_seconds=60,
        include_unsatisfiable=True,
        output_directory="sat_reasoning_dataset",
        random_seed=42
    )
    
    generator = DatasetGenerator(config)
    saved_files = generator.generate_and_save(["json", "jsonl"])
    
    print("Dataset generation complete!")
    print(f"Saved files: {saved_files}")