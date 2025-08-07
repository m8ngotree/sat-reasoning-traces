from typing import List, Dict, Set, Optional, Tuple, Any
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


class CDCLSolver:
    def __init__(self, instance: SATInstance):
        self.instance = instance
        self.assignments: Dict[int, bool] = {}
        self.decision_level = 0
        self.assignment_stack: List[Assignment] = []
        self.trace = SolverTrace(instance)
        self.step_counter = 0
        
        # For conflict analysis
        self.implication_graph: Dict[int, Set[int]] = {}
        
        # Watched literals for efficient unit propagation
        self.watched_literals: Dict[int, List[Clause]] = {}
        self._initialize_watched_literals()
    
    def _initialize_watched_literals(self):
        for var in range(1, self.instance.num_variables + 1):
            self.watched_literals[var] = []
            self.watched_literals[-var] = []
        
        for clause in self.instance.clauses:
            if len(clause.literals) >= 2:
                lit1, lit2 = clause.literals[0], clause.literals[1]
                self.watched_literals[lit1].append(clause)
                self.watched_literals[lit2].append(clause)
            elif len(clause.literals) == 1:
                lit = clause.literals[0]
                self.watched_literals[lit].append(clause)
    
    def _is_clause_satisfied(self, clause: Clause) -> bool:
        for lit in clause.literals:
            var = abs(lit)
            if var in self.assignments:
                expected_value = lit > 0
                if self.assignments[var] == expected_value:
                    return True
        return False
    
    def _is_clause_unit(self, clause: Clause) -> Tuple[bool, Optional[int]]:
        unassigned_literals = []
        satisfied = False
        
        for lit in clause.literals:
            var = abs(lit)
            if var in self.assignments:
                expected_value = lit > 0
                if self.assignments[var] == expected_value:
                    satisfied = True
                    break
            else:
                unassigned_literals.append(lit)
        
        if satisfied:
            return False, None
        
        if len(unassigned_literals) == 1:
            return True, unassigned_literals[0]
        
        return False, None
    
    def _propagate_unit_clauses(self) -> Optional[Clause]:
        changed = True
        
        while changed:
            changed = False
            assignments_before = copy.deepcopy(self.assignments)
            
            for clause in self.instance.clauses:
                if self._is_clause_satisfied(clause):
                    continue
                
                is_unit, unit_literal = self._is_clause_unit(clause)
                
                if is_unit and unit_literal is not None:
                    var = abs(unit_literal)
                    value = unit_literal > 0
                    
                    if var not in self.assignments:
                        assignment = Assignment(var, value, self.decision_level, clause)
                        self.assignments[var] = value
                        self.assignment_stack.append(assignment)
                        changed = True
                        
                        step = SolverStep(
                            step_number=self.step_counter,
                            decision_type=DecisionType.PROPAGATE,
                            assignment=assignment,
                            clause=clause,
                            assignments_before=assignments_before,
                            assignments_after=copy.deepcopy(self.assignments),
                            decision_level=self.decision_level,
                            explanation=f"Unit propagation: clause {clause} forces {assignment.variable} = {assignment.value}"
                        )
                        self.trace.add_step(step)
                        self.step_counter += 1
                    
                    elif self.assignments[var] != value:
                        # Conflict detected
                        step = SolverStep(
                            step_number=self.step_counter,
                            decision_type=DecisionType.CONFLICT,
                            assignment=None,
                            clause=clause,
                            assignments_before=assignments_before,
                            assignments_after=copy.deepcopy(self.assignments),
                            decision_level=self.decision_level,
                            explanation=f"Conflict detected: clause {clause} requires {var} = {value}, but {var} = {self.assignments[var]}"
                        )
                        self.trace.add_step(step)
                        self.step_counter += 1
                        return clause
        
        return None
    
    def _find_unassigned_variable(self) -> Optional[int]:
        for var in range(1, self.instance.num_variables + 1):
            if var not in self.assignments:
                return var
        return None
    
    def _make_decision(self, variable: int, value: bool):
        self.decision_level += 1
        assignments_before = copy.deepcopy(self.assignments)
        
        assignment = Assignment(variable, value, self.decision_level)
        self.assignments[variable] = value
        self.assignment_stack.append(assignment)
        
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.DECIDE,
            assignment=assignment,
            clause=None,
            assignments_before=assignments_before,
            assignments_after=copy.deepcopy(self.assignments),
            decision_level=self.decision_level,
            explanation=f"Decision: assign {variable} = {value} at level {self.decision_level}"
        )
        self.trace.add_step(step)
        self.step_counter += 1
    
    def _backtrack(self, target_level: int):
        assignments_before = copy.deepcopy(self.assignments)
        backtracked_vars = []
        
        while self.assignment_stack and self.assignment_stack[-1].decision_level > target_level:
            assignment = self.assignment_stack.pop()
            del self.assignments[assignment.variable]
            backtracked_vars.append(assignment.variable)
        
        self.decision_level = target_level
        
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.BACKTRACK,
            assignment=None,
            clause=None,
            assignments_before=assignments_before,
            assignments_after=copy.deepcopy(self.assignments),
            decision_level=self.decision_level,
            explanation=f"Backtrack to level {target_level}, unassigning variables: {backtracked_vars}"
        )
        self.trace.add_step(step)
        self.step_counter += 1
    
    def _analyze_conflict(self, conflict_clause: Clause) -> int:
        # Simple backtracking strategy - go back one level
        if self.decision_level > 0:
            return self.decision_level - 1
        return -1  # UNSAT
    
    def solve(self) -> bool:
        while True:
            # Unit propagation
            conflict_clause = self._propagate_unit_clauses()
            
            if conflict_clause is not None:
                # Conflict occurred
                backtrack_level = self._analyze_conflict(conflict_clause)
                
                if backtrack_level < 0:
                    # UNSAT
                    self.trace.final_result = False
                    self.trace.final_assignment = None
                    return False
                
                self._backtrack(backtrack_level)
                continue
            
            # All clauses satisfied?
            all_satisfied = True
            for clause in self.instance.clauses:
                if not self._is_clause_satisfied(clause):
                    all_satisfied = False
                    break
            
            if all_satisfied:
                # SAT
                self.trace.final_result = True
                self.trace.final_assignment = copy.deepcopy(self.assignments)
                return True
            
            # Make a decision
            unassigned_var = self._find_unassigned_variable()
            if unassigned_var is None:
                # All variables assigned but not all clauses satisfied - should not happen
                self.trace.final_result = False
                self.trace.final_assignment = None
                return False
            
            # Simple decision heuristic - try True first
            self._make_decision(unassigned_var, True)
    
    def get_trace(self) -> SolverTrace:
        return self.trace


class DPLLSolver:
    def __init__(self, instance: SATInstance):
        self.instance = instance
        self.trace = SolverTrace(instance)
        self.step_counter = 0
    
    def _dpll_recursive(self, clauses: List[Clause], assignments: Dict[int, bool], 
                       level: int) -> Tuple[bool, Optional[Dict[int, bool]]]:
        assignments_before = copy.deepcopy(assignments)
        
        # Simplify clauses based on current assignments
        simplified_clauses = []
        for clause in clauses:
            satisfied = False
            new_literals = []
            
            for lit in clause.literals:
                var = abs(lit)
                if var in assignments:
                    expected_value = lit > 0
                    if assignments[var] == expected_value:
                        satisfied = True
                        break
                    # Literal is false, don't add it
                else:
                    new_literals.append(lit)
            
            if satisfied:
                continue  # Clause is satisfied, ignore it
            
            if not new_literals:
                # Empty clause - conflict
                step = SolverStep(
                    step_number=self.step_counter,
                    decision_type=DecisionType.CONFLICT,
                    assignment=None,
                    clause=clause,
                    assignments_before=assignments_before,
                    assignments_after=copy.deepcopy(assignments),
                    decision_level=level,
                    explanation=f"Conflict: clause {clause} becomes empty"
                )
                self.trace.add_step(step)
                self.step_counter += 1
                return False, None
            
            simplified_clauses.append(Clause(new_literals))
        
        if not simplified_clauses:
            # All clauses satisfied
            return True, assignments
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            assignments_before = copy.deepcopy(assignments)
            
            for clause in simplified_clauses:
                if len(clause.literals) == 1:
                    lit = clause.literals[0]
                    var = abs(lit)
                    value = lit > 0
                    
                    if var not in assignments:
                        assignments[var] = value
                        assignment = Assignment(var, value, level, clause)
                        
                        step = SolverStep(
                            step_number=self.step_counter,
                            decision_type=DecisionType.PROPAGATE,
                            assignment=assignment,
                            clause=clause,
                            assignments_before=assignments_before,
                            assignments_after=copy.deepcopy(assignments),
                            decision_level=level,
                            explanation=f"Unit clause {clause} forces {var} = {value}"
                        )
                        self.trace.add_step(step)
                        self.step_counter += 1
                        changed = True
            
            # Re-simplify after unit propagation
            if changed:
                return self._dpll_recursive(clauses, assignments, level)
        
        # Choose a variable to branch on
        unassigned_vars = set(range(1, self.instance.num_variables + 1)) - set(assignments.keys())
        if not unassigned_vars:
            return True, assignments
        
        var = min(unassigned_vars)  # Simple heuristic
        
        # Try True first
        assignments_true = copy.deepcopy(assignments)
        assignments_true[var] = True
        assignment_true = Assignment(var, True, level + 1)
        
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.DECIDE,
            assignment=assignment_true,
            clause=None,
            assignments_before=assignments_before,
            assignments_after=copy.deepcopy(assignments_true),
            decision_level=level + 1,
            explanation=f"Decision: try {var} = True at level {level + 1}"
        )
        self.trace.add_step(step)
        self.step_counter += 1
        
        sat, solution = self._dpll_recursive(clauses, assignments_true, level + 1)
        if sat:
            return True, solution
        
        # Try False - use same decision level since this is part of the same branching
        assignments_false = copy.deepcopy(assignments)
        assignments_false[var] = False
        assignment_false = Assignment(var, False, level + 1)
        
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.DECIDE,
            assignment=assignment_false,
            clause=None,
            assignments_before=assignments_before,
            assignments_after=copy.deepcopy(assignments_false),
            decision_level=level + 1,
            explanation=f"Backtrack and try {var} = False at level {level + 1}"
        )
        self.trace.add_step(step)
        self.step_counter += 1
        
        return self._dpll_recursive(clauses, assignments_false, level + 1)
    
    def solve(self) -> bool:
        sat, solution = self._dpll_recursive(self.instance.clauses, {}, 0)
        self.trace.final_result = sat
        self.trace.final_assignment = solution
        return sat
    
    def get_trace(self) -> SolverTrace:
        return self.trace