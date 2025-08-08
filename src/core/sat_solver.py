from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
from .sat_generator import SATInstance, Clause


class DecisionType(Enum):
    DECIDE = "decide"
    PROPAGATE = "propagate" 
    CONFLICT = "conflict"
    BACKTRACK = "backtrack"
    RESTART = "restart"


@dataclass
class Assignment:
    variable: int
    value: bool
    decision_level: int
    reason: Optional[Clause] = None  # Clause that forced this assignment
    
    def __str__(self):
        sign = "" if self.value else "¬"
        reason_str = f" (reason: {self.reason})" if self.reason else ""
        return f"{sign}x{self.variable}@{self.decision_level}{reason_str}"


@dataclass
class SolverStep:
    step_number: int
    decision_type: DecisionType
    assignment: Optional[Assignment]
    clause: Optional[Clause]
    assignments_before: Dict[int, bool]
    assignments_after: Dict[int, bool]
    decision_level: int
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "decision_type": self.decision_type.value,
            "assignment": str(self.assignment) if self.assignment else None,
            "clause": str(self.clause) if self.clause else None,
            "assignments_before": dict(self.assignments_before),
            "assignments_after": dict(self.assignments_after),
            "decision_level": self.decision_level,
            "explanation": self.explanation
        }


@dataclass
class SolverTrace:
    instance: SATInstance
    steps: List[SolverStep] = field(default_factory=list)
    final_result: Optional[bool] = None
    final_assignment: Optional[Dict[int, bool]] = None
    
    def add_step(self, step: SolverStep):
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.instance.problem_type.value,
            "num_variables": self.instance.num_variables,
            "num_clauses": len(self.instance.clauses),
            "metadata": self.instance.metadata,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result,
            "final_assignment": self.final_assignment,
            "total_steps": len(self.steps)
        }



class DPLLSolver:
    def __init__(self, instance: SATInstance):
        self.instance = instance
        self.trace = SolverTrace(instance)
        self.step_counter = 0
    
    
    def solve(self) -> bool:
        """Main iterative DPLL solver with proper state tracking"""
        assignments = {}
        level = 0
        
        while True:
            # Unit propagation phase
            while True:
                unit_clause = self._find_unit_clause(assignments)
                if not unit_clause:
                    break  # No more unit clauses
                
                var, value = unit_clause
                if var in assignments:
                    continue  # Already assigned
                
                assignments_before = copy.deepcopy(assignments)
                assignments[var] = value
                
                # Find the clause that forced this assignment
                forcing_clause = self._find_forcing_clause(var, value, assignments_before)
                assignment = Assignment(var, value, level, forcing_clause)
                
                step = SolverStep(
                    step_number=self.step_counter,
                    decision_type=DecisionType.PROPAGATE,
                    assignment=assignment,
                    clause=forcing_clause,
                    assignments_before=assignments_before,
                    assignments_after=copy.deepcopy(assignments),
                    decision_level=level,
                    explanation=f"Unit clause {forcing_clause} forces {var} = {value}"
                )
                self.trace.add_step(step)
                self.step_counter += 1
                
                # Check for conflict
                if self._has_conflict(assignments):
                    conflict_clause = self._find_conflict_clause(assignments)
                    step = SolverStep(
                        step_number=self.step_counter,
                        decision_type=DecisionType.CONFLICT,
                        assignment=None,
                        clause=conflict_clause,
                        assignments_before=copy.deepcopy(assignments),
                        assignments_after=copy.deepcopy(assignments),
                        decision_level=level,
                        explanation=f"Conflict: clause {conflict_clause} becomes empty"
                    )
                    self.trace.add_step(step)
                    self.step_counter += 1
                    
                    self.trace.final_result = False
                    self.trace.final_assignment = None
                    return False
            
            # Check if all clauses are satisfied
            if self._all_clauses_satisfied(assignments):
                self.trace.final_result = True
                self.trace.final_assignment = assignments
                return True
            
            # Choose next variable to decide
            unassigned_vars = set(range(1, self.instance.num_variables + 1)) - set(assignments.keys())
            if not unassigned_vars:
                # All variables assigned but not satisfied - shouldn't happen
                self.trace.final_result = False
                self.trace.final_assignment = None
                return False
            
            var = min(unassigned_vars)  # Simple heuristic
            level += 1
            
            # Make decision
            assignments_before = copy.deepcopy(assignments)
            assignments[var] = True
            assignment = Assignment(var, True, level)
            
            step = SolverStep(
                step_number=self.step_counter,
                decision_type=DecisionType.DECIDE,
                assignment=assignment,
                clause=None,
                assignments_before=assignments_before,
                assignments_after=copy.deepcopy(assignments),
                decision_level=level,
                explanation=f"Decision: try {var} = True at level {level}"
            )
            self.trace.add_step(step)
            self.step_counter += 1
    
    
    def _find_unit_clause(self, assignments: Dict[int, bool]) -> Optional[Tuple[int, bool]]:
        """Find a unit clause and return the variable and value to assign"""
        for clause in self.instance.clauses:
            unassigned_literals = []
            satisfied = False
            
            for lit in clause.literals:
                var = abs(lit)
                expected_value = lit > 0
                
                if var in assignments:
                    if assignments[var] == expected_value:
                        satisfied = True
                        break
                else:
                    unassigned_literals.append((var, expected_value))
            
            if not satisfied and len(unassigned_literals) == 1:
                var, value = unassigned_literals[0]
                return var, value
        
        return None
    
    def _find_forcing_clause(self, var: int, value: bool, assignments: Dict[int, bool]) -> Clause:
        """Find the clause that forces this variable assignment"""
        for clause in self.instance.clauses:
            unassigned_count = 0
            satisfied = False
            target_literal = None
            
            for lit in clause.literals:
                curr_var = abs(lit)
                expected_value = lit > 0
                
                if curr_var == var and expected_value == value:
                    target_literal = lit
                elif curr_var in assignments:
                    if assignments[curr_var] == expected_value:
                        satisfied = True
                        break
                else:
                    unassigned_count += 1
            
            if not satisfied and unassigned_count == 0 and target_literal is not None:
                return clause
        
        # Return first clause containing the literal if not found
        for clause in self.instance.clauses:
            for lit in clause.literals:
                if abs(lit) == var and (lit > 0) == value:
                    return clause
        
        return self.instance.clauses[0]  # Fallback
    
    def _has_conflict(self, assignments: Dict[int, bool]) -> bool:
        """Check if current assignments create a conflict (empty clause)"""
        for clause in self.instance.clauses:
            satisfied = False
            has_unassigned = False
            
            for lit in clause.literals:
                var = abs(lit)
                expected_value = lit > 0
                
                if var in assignments:
                    if assignments[var] == expected_value:
                        satisfied = True
                        break
                else:
                    has_unassigned = True
            
            if not satisfied and not has_unassigned:
                return True  # Empty clause - conflict
        
        return False
    
    def _find_conflict_clause(self, assignments: Dict[int, bool]) -> Clause:
        """Find the clause that causes the conflict"""
        for clause in self.instance.clauses:
            satisfied = False
            has_unassigned = False
            
            for lit in clause.literals:
                var = abs(lit)
                expected_value = lit > 0
                
                if var in assignments:
                    if assignments[var] == expected_value:
                        satisfied = True
                        break
                else:
                    has_unassigned = True
            
            if not satisfied and not has_unassigned:
                return clause
        
        return self.instance.clauses[0]  # Fallback
    
    def _all_clauses_satisfied(self, assignments: Dict[int, bool]) -> bool:
        """Check if all clauses are satisfied"""
        for clause in self.instance.clauses:
            satisfied = False
            
            for lit in clause.literals:
                var = abs(lit)
                expected_value = lit > 0
                
                if var in assignments and assignments[var] == expected_value:
                    satisfied = True
                    break
            
            if not satisfied:
                return False
        
        return True
    
    def get_trace(self) -> SolverTrace:
        return self.trace