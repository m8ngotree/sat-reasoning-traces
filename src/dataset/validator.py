import json
import jsonschema
from typing import List, Dict, Any, Set, Optional, Tuple
import re
from pathlib import Path
import logging
from dataclasses import dataclass
import statistics

from src.core.sat_generator import ProblemType, Clause, SATInstance
from src.core.sat_solver import DecisionType


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


class DatasetValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # JSON schema for validating dataset structure
        self.instance_schema = {
            "type": "object",
            "required": ["instance_id", "problem", "reasoning_trace", "solution", "step_by_step", "solver_type"],
            "properties": {
                "instance_id": {"type": "integer"},
                "solver_type": {"type": "string", "enum": ["DPLL"]},
                "generation_seed": {"type": "integer"},
                "problem": {
                    "type": "object",
                    "required": ["type", "num_variables", "num_clauses", "clauses"],
                    "properties": {
                        "type": {"type": "string"},
                        "num_variables": {"type": "integer", "minimum": 1},
                        "num_clauses": {"type": "integer", "minimum": 0},
                        "clauses": {"type": "array", "items": {"type": "string"}},
                        "dimacs": {"type": "string"},
                        "metadata": {"type": "object"}
                    }
                },
                "reasoning_trace": {"type": "string", "minLength": 100},
                "solution": {
                    "type": "object",
                    "required": ["satisfiable", "steps_taken"],
                    "properties": {
                        "satisfiable": {"type": ["boolean", "null"]},
                        "assignment": {"type": ["object", "null"]},
                        "steps_taken": {"type": "integer", "minimum": 0},
                        "conflicts_encountered": {"type": "integer", "minimum": 0},
                        "decisions_made": {"type": "integer", "minimum": 0}
                    }
                },
                "step_by_step": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["step_number", "decision_type", "decision_level", "explanation"],
                        "properties": {
                            "step_number": {"type": "integer", "minimum": 0},
                            "decision_type": {"type": "string"},
                            "assignment": {"type": ["string", "null"]},
                            "clause": {"type": ["string", "null"]},
                            "assignments_before": {"type": "object"},
                            "assignments_after": {"type": "object"},
                            "decision_level": {"type": "integer", "minimum": 0},
                            "explanation": {"type": "string", "minLength": 10}
                        }
                    }
                }
            }
        }
    
    def validate_dataset_structure(self, dataset: Dict[str, Any]) -> ValidationResult:
        """Validate the overall dataset structure"""
        errors = []
        warnings = []
        
        # Check top-level structure
        if "instances" not in dataset:
            errors.append("Dataset missing 'instances' field")
            return ValidationResult(False, errors, warnings, {})
        
        if not isinstance(dataset["instances"], list):
            errors.append("'instances' field must be a list")
            return ValidationResult(False, errors, warnings, {})
        
        if len(dataset["instances"]) == 0:
            errors.append("Dataset contains no instances")
            return ValidationResult(False, errors, warnings, {})
        
        # Validate each instance
        valid_instances = 0
        for i, instance in enumerate(dataset["instances"]):
            try:
                jsonschema.validate(instance, self.instance_schema)
                valid_instances += 1
            except jsonschema.ValidationError as e:
                errors.append(f"Instance {i}: {e.message}")
            except Exception as e:
                errors.append(f"Instance {i}: Unexpected validation error: {str(e)}")
        
        # Check if we have any valid instances
        if valid_instances == 0:
            errors.append("No valid instances found in dataset")
            return ValidationResult(False, errors, warnings, {})
        
        if valid_instances < len(dataset["instances"]):
            warnings.append(f"Only {valid_instances}/{len(dataset['instances'])} instances are valid")
        
        statistics = {
            "total_instances": len(dataset["instances"]),
            "valid_instances": valid_instances,
            "invalid_instances": len(dataset["instances"]) - valid_instances
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def validate_sat_logic(self, instance: Dict[str, Any]) -> ValidationResult:
        """Validate the SAT problem logic and solver trace consistency"""
        errors = []
        warnings = []
        
        try:
            # Parse problem data
            num_variables = instance["problem"]["num_variables"]
            clauses = instance["problem"]["clauses"]
            satisfiable = instance["solution"]["satisfiable"]
            final_assignment = instance["solution"]["assignment"]
            steps = instance["step_by_step"]
            
            # Validate variable numbering
            max_var_in_clauses = 0
            for clause_str in clauses:
                variables = self._extract_variables_from_clause(clause_str)
                if variables:
                    max_var_in_clauses = max(max_var_in_clauses, max(variables))
            
            if max_var_in_clauses > num_variables:
                errors.append(f"Clause references variable {max_var_in_clauses} but only {num_variables} variables declared")
            
            # Validate satisfiability claim
            if satisfiable is True and final_assignment:
                is_actually_sat = self._check_satisfiability(clauses, final_assignment)
                if not is_actually_sat:
                    errors.append("Instance claimed satisfiable but assignment doesn't satisfy all clauses")
            
            # Validate solver trace consistency
            trace_errors = self._validate_solver_trace(steps, num_variables)
            errors.extend(trace_errors)
            
            # Check for reasonable solving statistics
            decisions = sum(1 for step in steps if step.get("decision_type") == "decide")
            conflicts = sum(1 for step in steps if step.get("decision_type") == "conflict")
            propagations = sum(1 for step in steps if step.get("decision_type") == "propagate")
            backtracks = sum(1 for step in steps if step.get("decision_type") == "backtrack")
            
            # More sophisticated checks
            if satisfiable is False and conflicts == 0:
                warnings.append("Unsatisfiable instance with no conflicts recorded - may indicate solver implementation issues")
            
            if satisfiable is False and backtracks == 0 and conflicts > 0:
                warnings.append("Unsatisfiable instance with conflicts but no backtracking - unusual")
            
            if decisions > num_variables * 3:
                warnings.append(f"Very high number of decisions ({decisions}) for {num_variables} variables - may indicate inefficient solving")
            
            if propagations == 0 and len(steps) > 1:
                warnings.append("No unit propagation steps found - unusual for modern SAT solvers")
            
            # Check decision/conflict ratio
            if conflicts > 0 and decisions / conflicts > 10:
                warnings.append(f"High decision-to-conflict ratio ({decisions}/{conflicts}) - may indicate poor conflict analysis")
            
        except Exception as e:
            errors.append(f"Error during SAT logic validation: {str(e)}")
        
        statistics = {
            "max_variable": max_var_in_clauses if 'max_var_in_clauses' in locals() else 0,
            "decision_steps": decisions if 'decisions' in locals() else 0,
            "conflict_steps": conflicts if 'conflicts' in locals() else 0,
            "propagation_steps": propagations if 'propagations' in locals() else 0,
            "backtrack_steps": backtracks if 'backtracks' in locals() else 0
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def _extract_variables_from_clause(self, clause_str: str) -> Set[int]:
        """Extract variable numbers from a clause string"""
        variables = set()
        # Handle Unicode characters
        tokens = clause_str.replace("¬", "").replace("¬", "").replace("∨", "").replace("∨", "").replace("x", " ").split()
        for token in tokens:
            try:
                var = int(token.strip())
                if var > 0:
                    variables.add(var)
            except ValueError:
                continue
        return variables
    
    def _check_satisfiability(self, clauses: List[str], assignment: Dict[str, bool]) -> bool:
        """Check if an assignment satisfies all clauses"""
        # Convert string keys to integers
        int_assignment = {}
        for key, value in assignment.items():
            try:
                if isinstance(key, str):
                    int_assignment[int(key)] = value
                else:
                    int_assignment[key] = value
            except ValueError:
                continue
        
        for clause_str in clauses:
            if not self._is_clause_satisfied(clause_str, int_assignment):
                return False
        return True
    
    def _is_clause_satisfied(self, clause_str: str, assignment: Dict[int, bool]) -> bool:
        """Check if a single clause is satisfied by an assignment.
        Supports both Unicode (¬, ∨) and ASCII (NOT, OR) formats.
        """
        text = clause_str.strip()
        # Choose splitter: prefer Unicode disjunction, otherwise ASCII OR (case-insensitive)
        if "∨" in text:
            parts = [p.strip() for p in text.split("∨") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"\bOR\b", text, flags=re.IGNORECASE) if p.strip()]

        for literal in parts:
            lit = literal.strip()
            # Detect negation via Unicode or ASCII
            is_neg = ("¬" in lit) or (re.search(r"\bNOT\b", lit, flags=re.IGNORECASE) is not None)

            # Extract variable number like x12 or X12
            m = re.search(r"[xX]\s*(\d+)", lit)
            if not m:
                # Fallback: try to parse any trailing number token
                m2 = re.search(r"(\d+)", lit)
                if not m2:
                    continue
                var_num = int(m2.group(1))
            else:
                var_num = int(m.group(1))

            if var_num in assignment:
                var_value = assignment[var_num]
                val = var_value if not is_neg else (not var_value)
                if val:
                    return True

        return False  # No literal satisfied the clause
    
    def _validate_solver_trace(self, steps: List[Dict[str, Any]], num_variables: int) -> List[str]:
        """Validate the consistency of the solver trace with improved checks"""
        errors = []
        
        if not steps:
            errors.append("Empty solver trace")
            return errors
        
        # Track state for consistency checking
        current_assignments = {}
        current_decision_level = 0
        decision_count = 0
        conflict_count = 0
        propagation_count = 0
        
        # Check step numbering and trace consistency
        for i, step in enumerate(steps):
            decision_type = step.get("decision_type", "")
            step_level = step.get("decision_level", 0)
            assignments_before = step.get("assignments_before", {})
            assignments_after = step.get("assignments_after", {})
            
            # Validate step number
            if step.get("step_number", -1) != i:
                errors.append(f"Step {i}: Expected step number {i}, got {step.get('step_number')}")
            
            # Validate decision type
            if decision_type not in ["decide", "propagate", "conflict", "backtrack", "restart"]:
                errors.append(f"Step {i}: Invalid decision type '{decision_type}'")
                continue
            
            # Count step types
            if decision_type == "decide":
                decision_count += 1
            elif decision_type == "conflict":
                conflict_count += 1
            elif decision_type == "propagate":
                propagation_count += 1
            
            # Validate decision level consistency
            if decision_type == "decide":
                if step_level != current_decision_level + 1:
                    errors.append(f"Step {i}: Decision should increment level to {current_decision_level + 1}, got {step_level}")
                current_decision_level = step_level
            elif decision_type == "backtrack":
                if step_level >= current_decision_level:
                    errors.append(f"Step {i}: Backtrack should decrease level below {current_decision_level}, got {step_level}")
                current_decision_level = step_level
            elif decision_type == "propagate":
                if step_level != current_decision_level:
                    errors.append(f"Step {i}: Propagation should be at current level {current_decision_level}, got {step_level}")
            
            # Validate assignment consistency
            if i == 0:
                # First step should have empty assignments_before
                if assignments_before:
                    errors.append(f"Step {i}: First step should have empty assignments_before")
            else:
                # Check that assignments_before matches previous step's assignments_after
                prev_step = steps[i-1]
                prev_assignments_after = prev_step.get("assignments_after", {})
                if assignments_before != prev_assignments_after:
                    errors.append(f"Step {i}: assignments_before doesn't match previous step's assignments_after")
            
            # Validate variable bounds in assignments
            for var_str in list(assignments_before.keys()) + list(assignments_after.keys()):
                try:
                    var = int(var_str)
                    if var < 1 or var > num_variables:
                        errors.append(f"Step {i}: Variable {var} out of bounds (1-{num_variables})")
                except ValueError:
                    errors.append(f"Step {i}: Invalid variable identifier '{var_str}'")
            
            # For decision steps, check that exactly one variable is assigned
            if decision_type == "decide":
                diff_vars = set(assignments_after.keys()) - set(assignments_before.keys())
                if len(diff_vars) != 1:
                    errors.append(f"Step {i}: Decision should assign exactly one variable, assigned {len(diff_vars)}")
            
            # For propagation, check that assignment is justified
            elif decision_type == "propagate":
                diff_vars = set(assignments_after.keys()) - set(assignments_before.keys())
                if len(diff_vars) != 1:
                    errors.append(f"Step {i}: Propagation should assign exactly one variable, assigned {len(diff_vars)}")
                
                # Check that a clause is provided as justification
                if not step.get("clause"):
                    errors.append(f"Step {i}: Propagation step missing justifying clause")
            
            # Update current state
            current_assignments = dict(assignments_after)
        
        # Validate overall trace statistics
        # High decision count is not inherently incorrect; surface as a warning in higher-level checks
        
        # Check that unsatisfiable instances have conflicts
        has_final_result = len(steps) > 0 and steps[-1].get("decision_type") in ["conflict", "backtrack"]
        if conflict_count == 0 and not has_final_result:
            # This will be flagged as a warning in the main validation, not an error
            pass
        
        return errors
    
    def validate_reasoning_quality(self, instance: Dict[str, Any]) -> ValidationResult:
        """Validate the quality of natural language reasoning traces"""
        errors = []
        warnings = []
        
        reasoning_trace = instance.get("reasoning_trace", "")
        
        # Check minimum length
        if len(reasoning_trace) < 500:
            warnings.append("Reasoning trace is quite short (< 500 characters)")

        # Support both legacy sectioned format and new tag-structured format
        has_tagged = "<final_answer>" in reasoning_trace
        if has_tagged:
            # Tagged format checks
            tag_requirements = ["<problem>", "<steps>", "<final_answer>"]
            for tag in tag_requirements:
                if tag not in reasoning_trace:
                    errors.append(f"Missing tag: '{tag}' in reasoning trace")
        else:
            # Legacy header-based format checks
            required_sections = ["Problem Description", "Solving Process", "Step-by-Step Reasoning", "Final Result"]
            for section in required_sections:
                if section not in reasoning_trace:
                    errors.append(f"Missing section: '{section}' in reasoning trace")
        
        # Check explanation quality in steps
        steps = instance.get("step_by_step", [])
        short_explanations = 0
        
        for i, step in enumerate(steps):
            explanation = step.get("explanation", "")
            if len(explanation) < 20:
                short_explanations += 1
        
        if short_explanations > len(steps) * 0.3:  # More than 30% have short explanations
            warnings.append(f"Many steps have short explanations ({short_explanations}/{len(steps)})")
        
        # Check for mathematical notation (support Unicode or ASCII)
        has_unicode = ("∨" in reasoning_trace) or ("∧" in reasoning_trace)
        has_ascii = re.search(r"\b(OR|AND)\b", reasoning_trace, flags=re.IGNORECASE) is not None
        if not (has_unicode or has_ascii):
            warnings.append("Reasoning trace lacks mathematical notation")
        
        statistics = {
            "trace_length": len(reasoning_trace),
            "num_steps": len(steps),
            "short_explanations": short_explanations
        }
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def validate_complete_dataset(self, dataset: Dict[str, Any]) -> ValidationResult:
        """Run complete validation on a dataset"""
        all_errors = []
        all_warnings = []
        all_statistics = {}
        
        # Structure validation
        struct_result = self.validate_dataset_structure(dataset)
        all_errors.extend([f"Structure: {e}" for e in struct_result.errors])
        all_warnings.extend([f"Structure: {w}" for w in struct_result.warnings])
        all_statistics.update(struct_result.statistics)
        
        if not struct_result.is_valid:
            return ValidationResult(False, all_errors, all_warnings, all_statistics)
        
        # Instance-level validation
        valid_instances = 0
        logic_errors = 0
        quality_errors = 0
        
        for i, instance in enumerate(dataset["instances"][:100]):  # Validate first 100 for performance
            # SAT logic validation
            logic_result = self.validate_sat_logic(instance)
            if logic_result.is_valid:
                valid_instances += 1
            else:
                logic_errors += 1
                all_errors.extend([f"Instance {i} Logic: {e}" for e in logic_result.errors])
            
            all_warnings.extend([f"Instance {i} Logic: {w}" for w in logic_result.warnings])
            
            # Reasoning quality validation
            quality_result = self.validate_reasoning_quality(instance)
            if not quality_result.is_valid:
                quality_errors += 1
                all_errors.extend([f"Instance {i} Quality: {e}" for e in quality_result.errors])
            
            all_warnings.extend([f"Instance {i} Quality: {w}" for w in quality_result.warnings])
        
        # Overall statistics
        all_statistics.update({
            "validation_sample_size": min(100, len(dataset["instances"])),
            "logically_valid_instances": valid_instances,
            "logic_errors": logic_errors,
            "quality_errors": quality_errors
        })
        
        # Dataset-level statistics
        problem_types = {}
        satisfiability_dist = {"sat": 0, "unsat": 0, "unknown": 0}
        # Per-problem-type satisfiability breakdown
        satisfiability_by_problem_type: Dict[str, Dict[str, int]] = {}
        
        for instance in dataset["instances"]:
            prob_type = instance.get("problem", {}).get("type", "unknown")
            problem_types[prob_type] = problem_types.get(prob_type, 0) + 1
            
            sat_result = instance.get("solution", {}).get("satisfiable")
            if sat_result is True:
                satisfiability_dist["sat"] += 1
                satisfiability_by_problem_type.setdefault(prob_type, {"sat": 0, "unsat": 0, "unknown": 0})["sat"] += 1
            elif sat_result is False:
                satisfiability_dist["unsat"] += 1
                satisfiability_by_problem_type.setdefault(prob_type, {"sat": 0, "unsat": 0, "unknown": 0})["unsat"] += 1
            else:
                satisfiability_dist["unknown"] += 1
                satisfiability_by_problem_type.setdefault(prob_type, {"sat": 0, "unsat": 0, "unknown": 0})["unknown"] += 1
        
        all_statistics.update({
            "problem_type_distribution": problem_types,
            "satisfiability_distribution": satisfiability_dist,
            "satisfiability_by_problem_type": satisfiability_by_problem_type
        })
        
        is_valid = len(all_errors) == 0
        return ValidationResult(is_valid, all_errors, all_warnings, all_statistics)
    
    def validate_dataset_file(self, file_path: Path) -> ValidationResult:
        """Validate a dataset file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    dataset = json.load(f)
                elif file_path.suffix == '.jsonl':
                    # Load JSONL format
                    instances = []
                    for line in f:
                        if line.strip():
                            instances.append(json.loads(line))
                    dataset = {"instances": instances}
                else:
                    return ValidationResult(False, [f"Unsupported file format: {file_path.suffix}"], [], {})
            
            return self.validate_complete_dataset(dataset)
            
        except json.JSONDecodeError as e:
            return ValidationResult(False, [f"JSON parsing error: {str(e)}"], [], {})
        except Exception as e:
            return ValidationResult(False, [f"File reading error: {str(e)}"], [], {})
    
    def generate_validation_report(self, result: ValidationResult, output_path: Optional[Path] = None) -> str:
        """Generate a human-readable validation report"""
        report = "# Dataset Validation Report\n\n"
        
        # Overall status
        status = "VALID" if result.is_valid else "INVALID"
        report += f"**Overall Status**: {status}\n\n"
        
        # Statistics
        if result.statistics:
            report += "## Statistics\n"

            def _format_dict(d: Dict[str, Any], indent: int = 0) -> str:
                lines = []
                pad = "  " * indent
                for k, v in d.items():
                    if isinstance(v, dict):
                        lines.append(f"{pad}- {k}:")
                        lines.append(_format_dict(v, indent + 1))
                    else:
                        lines.append(f"{pad}- {k}: {v}")
                return "\n".join(lines)

            for key, value in result.statistics.items():
                if isinstance(value, dict):
                    report += f"- **{key}**:\n"
                    report += _format_dict(value, indent=1) + "\n"
                else:
                    report += f"- **{key}**: {value}\n"
            report += "\n"
        
        # Errors
        if result.errors:
            report += f"## Errors ({len(result.errors)})\n"
            for error in result.errors:
                report += f"- {error}\n"
            report += "\n"
        
        # Warnings
        if result.warnings:
            report += f"## Warnings ({len(result.warnings)})\n"
            for warning in result.warnings:
                report += f"- {warning}\n"
            report += "\n"
        
        if not result.errors and not result.warnings:
            report += "## No Issues Found\n"
            report += "The dataset passed all validation checks.\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


if __name__ == "__main__":
    # Example validation
    validator = DatasetValidator()
    
    # Validate a dataset file
    dataset_path = Path("sat_reasoning_dataset/sat_dataset_20240101_120000.json")
    if dataset_path.exists():
        result = validator.validate_dataset_file(dataset_path)
        report = validator.generate_validation_report(result)
        print(report)
    else:
        print("No dataset file found for validation")