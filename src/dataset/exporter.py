import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import torch
import pickle
import h5py
import numpy as np
from datetime import datetime


@dataclass 
class ExportConfig:
    output_directory: str = "exports"
    include_reasoning_traces: bool = True
    include_step_details: bool = True
    max_trace_length: Optional[int] = None
    anonymize_data: bool = False


class DatasetExporter:
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def export_to_huggingface_format(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset in Hugging Face datasets format"""
        hf_data = []
        
        for instance in dataset["instances"]:
            # Create instruction-response pairs for training
            problem_desc = self._create_problem_instruction(instance)
            reasoning_trace = instance["reasoning_trace"]
            
            # Truncate if needed
            if self.config.max_trace_length:
                reasoning_trace = reasoning_trace[:self.config.max_trace_length]
            
            hf_instance = {
                "instruction": problem_desc,
                "response": reasoning_trace,
                "problem_type": instance["problem"]["type"],
                "num_variables": instance["problem"]["num_variables"],
                "num_clauses": instance["problem"]["num_clauses"],
                "satisfiable": instance["solution"]["satisfiable"],
                "solver_type": instance["solver_type"],
                "steps_taken": instance["solution"]["steps_taken"],
                "conflicts": instance["solution"]["conflicts_encountered"],
                "decisions": instance["solution"]["decisions_made"]
            }
            
            # Add step details if requested
            if self.config.include_step_details:
                hf_instance["step_by_step"] = instance["step_by_step"]
            
            hf_data.append(hf_instance)
        
        # Save in JSONL format (common for HF datasets)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_hf_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in hf_data:
                f.write(json.dumps(item) + '\n')
        
        # Create dataset info file
        info = {
            "dataset_name": dataset_name,
            "description": "SAT reasoning traces for logical reasoning training",
            "num_instances": len(hf_data),
            "format": "instruction_response",
            "created": timestamp,
            "features": {
                "instruction": "Problem description and solving task",
                "response": "Detailed reasoning trace with natural language explanations",
                "problem_type": "Type of SAT problem (random_3sat, pigeonhole, etc.)",
                "num_variables": "Number of boolean variables in the problem",
                "num_clauses": "Number of clauses in the SAT formula",
                "satisfiable": "Whether the problem is satisfiable (boolean)",
                "solver_type": "SAT solver used (DPLL only)",
                "steps_taken": "Number of solver steps",
                "conflicts": "Number of conflicts encountered",
                "decisions": "Number of decision points"
            }
        }
        
        info_file = self.output_dir / f"{dataset_name}_hf_info_{timestamp}.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Exported HuggingFace format to {output_file}")
        return output_file
    
    def export_to_openai_format(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset in OpenAI fine-tuning format"""
        openai_data = []
        
        for instance in dataset["instances"]:
            problem_desc = self._create_problem_instruction(instance)
            reasoning_trace = instance["reasoning_trace"]
            
            if self.config.max_trace_length:
                reasoning_trace = reasoning_trace[:self.config.max_trace_length]
            
            openai_instance = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in Boolean satisfiability (SAT) solving. Provide detailed step-by-step reasoning when solving SAT problems."
                    },
                    {
                        "role": "user", 
                        "content": problem_desc
                    },
                    {
                        "role": "assistant",
                        "content": reasoning_trace
                    }
                ]
            }
            
            openai_data.append(openai_instance)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_openai_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in openai_data:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Exported OpenAI format to {output_file}")
        return output_file
    
    def export_to_alpaca_format(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset in Alpaca instruction-following format"""
        alpaca_data = []
        
        for instance in dataset["instances"]:
            problem_desc = self._create_problem_instruction(instance)
            reasoning_trace = instance["reasoning_trace"]
            
            if self.config.max_trace_length:
                reasoning_trace = reasoning_trace[:self.config.max_trace_length]
            
            alpaca_instance = {
                "instruction": "Solve the following Boolean satisfiability (SAT) problem with detailed reasoning.",
                "input": problem_desc,
                "output": reasoning_trace
            }
            
            alpaca_data.append(alpaca_instance)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_alpaca_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(alpaca_data, f, indent=2)
        
        self.logger.info(f"Exported Alpaca format to {output_file}")
        return output_file
    
    def export_to_csv(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset to CSV format for analysis"""
        csv_data = []
        
        for instance in dataset["instances"]:
            row = {
                "instance_id": instance["instance_id"],
                "problem_type": instance["problem"]["type"],
                "num_variables": instance["problem"]["num_variables"],
                "num_clauses": instance["problem"]["num_clauses"],
                "satisfiable": instance["solution"]["satisfiable"],
                "solver_type": instance["solver_type"],
                "steps_taken": instance["solution"]["steps_taken"],
                "conflicts_encountered": instance["solution"]["conflicts_encountered"],
                "decisions_made": instance["solution"]["decisions_made"],
                "clause_to_var_ratio": instance["problem"]["num_clauses"] / instance["problem"]["num_variables"]
            }
            
            # Add metadata fields
            metadata = instance["problem"].get("metadata", {})
            for key, value in metadata.items():
                if isinstance(value, (int, float, bool, str)):
                    row[f"metadata_{key}"] = value
            
            # Add reasoning trace if requested (truncated)
            if self.config.include_reasoning_traces:
                trace = instance["reasoning_trace"]
                if self.config.max_trace_length:
                    trace = trace[:self.config.max_trace_length]
                row["reasoning_trace"] = trace.replace('\n', ' ').replace('\r', ' ')
            
            csv_data.append(row)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_{timestamp}.csv"
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Exported CSV format to {output_file}")
        return output_file
    
    def export_to_pytorch(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset as PyTorch tensors"""
        # Extract numerical features
        features = []
        labels = []
        metadata = []
        
        for instance in dataset["instances"]:
            feature_vector = [
                instance["problem"]["num_variables"],
                instance["problem"]["num_clauses"],
                instance["problem"]["num_clauses"] / instance["problem"]["num_variables"],  # ratio
                instance["solution"]["steps_taken"],
                instance["solution"]["conflicts_encountered"],
                instance["solution"]["decisions_made"],
                1,  # solver type encoding (always DPLL)
                hash(instance["problem"]["type"]) % 1000,  # problem type hash
            ]
            
            features.append(feature_vector)
            labels.append(1 if instance["solution"]["satisfiable"] else 0)
            
            # Keep text data separately
            metadata.append({
                "instance_id": instance["instance_id"],
                "reasoning_trace": instance["reasoning_trace"],
                "clauses": instance["problem"]["clauses"],
                "step_by_step": instance["step_by_step"] if self.config.include_step_details else None
            })
        
        # Convert to tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_pytorch_{timestamp}.pt"
        
        torch.save({
            "features": feature_tensor,
            "labels": label_tensor,
            "metadata": metadata,
            "feature_names": [
                "num_variables", "num_clauses", "clause_var_ratio", 
                "steps_taken", "conflicts", "decisions", 
                "solver_cdcl", "problem_type_hash"
            ]
        }, output_file)
        
        self.logger.info(f"Exported PyTorch format to {output_file}")
        return output_file
    
    def export_to_hdf5(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset to HDF5 format for large-scale storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_{timestamp}.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            problems_group = f.create_group("problems")
            solutions_group = f.create_group("solutions")
            traces_group = f.create_group("traces")
            
            # Prepare arrays
            num_instances = len(dataset["instances"])
            
            # Problem data
            problem_types = []
            num_variables = np.zeros(num_instances, dtype=np.int32)
            num_clauses = np.zeros(num_instances, dtype=np.int32)
            
            # Solution data
            satisfiable = np.zeros(num_instances, dtype=bool)
            steps_taken = np.zeros(num_instances, dtype=np.int32)
            conflicts = np.zeros(num_instances, dtype=np.int32)
            decisions = np.zeros(num_instances, dtype=np.int32)
            solver_types = []
            
            # Traces (variable length strings)
            reasoning_traces = []
            
            for i, instance in enumerate(dataset["instances"]):
                problem_types.append(instance["problem"]["type"])
                num_variables[i] = instance["problem"]["num_variables"]
                num_clauses[i] = instance["problem"]["num_clauses"]
                
                satisfiable[i] = instance["solution"]["satisfiable"] if instance["solution"]["satisfiable"] is not None else False
                steps_taken[i] = instance["solution"]["steps_taken"]
                conflicts[i] = instance["solution"]["conflicts_encountered"]
                decisions[i] = instance["solution"]["decisions_made"]
                solver_types.append(instance["solver_type"])
                
                trace = instance["reasoning_trace"]
                if self.config.max_trace_length:
                    trace = trace[:self.config.max_trace_length]
                reasoning_traces.append(trace.encode('utf-8'))
            
            # Store arrays
            problems_group.create_dataset("types", data=[s.encode('utf-8') for s in problem_types])
            problems_group.create_dataset("num_variables", data=num_variables)
            problems_group.create_dataset("num_clauses", data=num_clauses)
            
            solutions_group.create_dataset("satisfiable", data=satisfiable)
            solutions_group.create_dataset("steps_taken", data=steps_taken)
            solutions_group.create_dataset("conflicts", data=conflicts)
            solutions_group.create_dataset("decisions", data=decisions)
            solutions_group.create_dataset("solver_types", data=[s.encode('utf-8') for s in solver_types])
            
            # Variable length strings for traces (modern dtype)
            dt = h5py.string_dtype(encoding='utf-8')
            traces_group.create_dataset("reasoning_traces", data=[t.decode('utf-8') for t in reasoning_traces], dtype=dt)
            
            # Store metadata
            f.attrs["num_instances"] = num_instances
            f.attrs["created"] = timestamp
            f.attrs["dataset_name"] = dataset_name
        
        self.logger.info(f"Exported HDF5 format to {output_file}")
        return output_file
    
    def export_to_xml(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Path:
        """Export dataset to XML format"""
        root = ET.Element("sat_reasoning_dataset")
        root.set("name", dataset_name)
        root.set("created", datetime.now().isoformat())
        root.set("num_instances", str(len(dataset["instances"])))
        
        for instance in dataset["instances"]:
            instance_elem = ET.SubElement(root, "instance")
            instance_elem.set("id", str(instance["instance_id"]))
            
            # Problem
            problem_elem = ET.SubElement(instance_elem, "problem")
            problem_elem.set("type", instance["problem"]["type"])
            problem_elem.set("variables", str(instance["problem"]["num_variables"]))
            problem_elem.set("clauses", str(instance["problem"]["num_clauses"]))
            
            # Clauses
            clauses_elem = ET.SubElement(problem_elem, "clauses")
            for i, clause in enumerate(instance["problem"]["clauses"]):
                clause_elem = ET.SubElement(clauses_elem, "clause")
                clause_elem.set("id", str(i))
                clause_elem.text = clause
            
            # Solution
            solution_elem = ET.SubElement(instance_elem, "solution")
            solution_elem.set("satisfiable", str(instance["solution"]["satisfiable"]))
            solution_elem.set("steps", str(instance["solution"]["steps_taken"]))
            solution_elem.set("conflicts", str(instance["solution"]["conflicts_encountered"]))
            solution_elem.set("decisions", str(instance["solution"]["decisions_made"]))
            solution_elem.set("solver", instance["solver_type"])
            
            # Reasoning trace
            if self.config.include_reasoning_traces:
                trace_elem = ET.SubElement(instance_elem, "reasoning_trace")
                trace = instance["reasoning_trace"]
                if self.config.max_trace_length:
                    trace = trace[:self.config.max_trace_length]
                trace_elem.text = trace
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_{timestamp}.xml"
        
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Exported XML format to {output_file}")
        return output_file
    
    def _create_problem_instruction(self, instance: Dict[str, Any]) -> str:
        """Create instruction text for the SAT problem"""
        problem = instance["problem"]
        
        instruction = f"Solve this {problem['type'].replace('_', ' ')} Boolean satisfiability problem:\n\n"
        instruction += f"Variables: {problem['num_variables']}\n"
        instruction += f"Clauses: {problem['num_clauses']}\n\n"
        
        # Show first few clauses
        instruction += "Clauses to satisfy:\n"
        for i, clause in enumerate(problem["clauses"][:10]):
            instruction += f"{i+1}. {clause}\n"
        
        if len(problem["clauses"]) > 10:
            instruction += f"... and {len(problem['clauses']) - 10} more clauses\n"
        
        instruction += "\nProvide step-by-step reasoning to determine if this formula is satisfiable."
        
        return instruction
    
    def export_all_formats(self, dataset: Dict[str, Any], dataset_name: str = "sat_reasoning") -> Dict[str, Path]:
        """Export dataset in all supported formats"""
        exported_files = {}
        
        try:
            exported_files["huggingface"] = self.export_to_huggingface_format(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"HuggingFace export failed: {str(e)}")
        
        try:
            exported_files["openai"] = self.export_to_openai_format(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"OpenAI export failed: {str(e)}")
        
        try:
            exported_files["alpaca"] = self.export_to_alpaca_format(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"Alpaca export failed: {str(e)}")
        
        try:
            exported_files["csv"] = self.export_to_csv(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"CSV export failed: {str(e)}")
        
        try:
            exported_files["pytorch"] = self.export_to_pytorch(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"PyTorch export failed: {str(e)}")
        
        try:
            exported_files["hdf5"] = self.export_to_hdf5(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"HDF5 export failed: {str(e)}")
        
        try:
            exported_files["xml"] = self.export_to_xml(dataset, dataset_name)
        except Exception as e:
            self.logger.error(f"XML export failed: {str(e)}")
        
        self.logger.info(f"Exported dataset in {len(exported_files)} formats")
        return exported_files


if __name__ == "__main__":
    # Example usage
    config = ExportConfig(
        output_directory="exports",
        include_reasoning_traces=True,
        include_step_details=False,
        max_trace_length=5000
    )
    
    exporter = DatasetExporter(config)
    
    # Load a dataset
    dataset_file = Path("sat_reasoning_dataset/sat_dataset_20240101_120000.json")
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Export all formats
        exported_files = exporter.export_all_formats(dataset, "sat_reasoning_v1")
        
        print("Export complete!")
        for format_name, file_path in exported_files.items():
            print(f"{format_name}: {file_path}")
    else:
        print("No dataset file found for export")