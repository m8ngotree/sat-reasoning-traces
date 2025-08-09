import unittest
from typing import Dict

from src.core.sat_generator import Clause, SATInstance, ProblemType, SATGenerator
from src.core.sat_solver import DPLLSolver, DecisionType


def is_clause_satisfied(clause: Clause, assignment: Dict[int, bool]) -> bool:
    for lit in clause.literals:
        var = abs(lit)
        expected = lit > 0
        if var in assignment and assignment[var] == expected:
            return True
    return False


def is_formula_satisfied(instance: SATInstance, assignment: Dict[int, bool]) -> bool:
    return all(is_clause_satisfied(c, assignment) for c in instance.clauses)


class TestDPLLSolverBasic(unittest.TestCase):
    def test_simple_sat(self):
        # (x1) AND (x2)
        inst = SATInstance(
            num_variables=2,
            clauses=[Clause([1]), Clause([2])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertTrue(result)
        self.assertIsNotNone(trace.final_assignment)
        self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))
        self.assertGreater(len(trace.steps), 0)

    def test_simple_unsat(self):
        # (x1) AND (¬x1)
        inst = SATInstance(
            num_variables=1,
            clauses=[Clause([1]), Clause([-1])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertFalse(result)
        self.assertIsNone(trace.final_assignment)
        self.assertGreater(len(trace.steps), 0)

    def test_pigeonhole_unsat_small(self):
        gen = SATGenerator(seed=0)
        inst = gen.generate_pigeonhole_principle(num_pigeons=3, num_holes=2)
        solver = DPLLSolver(inst)
        result = solver.solve()
        self.assertFalse(result)

    def test_trace_consistency(self):
        # A small SAT instance that triggers both decide and propagate
        # (x1 ∨ x2) AND (¬x1 ∨ x2) AND (x1)
        inst = SATInstance(
            num_variables=2,
            clauses=[Clause([1, 2]), Clause([-1, 2]), Clause([1])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        _ = solver.solve()
        steps = solver.get_trace().steps

        # assignments_before should equal previous assignments_after
        for i in range(1, len(steps)):
            self.assertEqual(steps[i].assignments_before, steps[i-1].assignments_after)

        # Decision levels: DECIDE should increment by 1 vs previous level
        prev_level = 0
        for step in steps:
            if step.decision_type == DecisionType.DECIDE:
                self.assertEqual(step.decision_level, prev_level + 1)
                prev_level = step.decision_level
            elif step.decision_type == DecisionType.BACKTRACK:
                self.assertLess(step.decision_level, prev_level)
                prev_level = step.decision_level


class TestDPLLSolverAdvanced(unittest.TestCase):
    def test_backtracking_needed_sat(self):
        # Construct a formula with no initial unit clauses where x1=True causes conflict,
        # forcing backtrack to x1=False which is SAT.
        # Clauses: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x1 ∨ ¬x3)
        inst = SATInstance(
            num_variables=3,
            clauses=[
                Clause([1, 2]),
                Clause([-1, 3]),
                Clause([-1, -3]),
            ],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={},
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertTrue(result)
        self.assertIsNotNone(trace.final_assignment)
        self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))
        # Ensure there was at least one DECIDE and a BACKTRACK before success
        decision_steps = [s for s in trace.steps if s.decision_type == DecisionType.DECIDE]
        backtracks = [s for s in trace.steps if s.decision_type == DecisionType.BACKTRACK]
        self.assertGreaterEqual(len(decision_steps), 2)
        self.assertGreaterEqual(len(backtracks), 1)

    def test_graph_coloring_triangle_sat(self):
        # Triangle graph with 3 colors is SAT
        gen = SATGenerator(seed=0)
        edges = [(0, 1), (1, 2), (0, 2)]
        inst = gen.generate_graph_coloring(num_vertices=3, edges=edges, num_colors=3)
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertTrue(result)
        self.assertIsNotNone(trace.final_assignment)
        self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))

    def test_graph_coloring_triangle_unsat(self):
        # Triangle graph with 2 colors is UNSAT
        gen = SATGenerator(seed=0)
        edges = [(0, 1), (1, 2), (0, 2)]
        inst = gen.generate_graph_coloring(num_vertices=3, edges=edges, num_colors=2)
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertFalse(result)
        self.assertIsNone(trace.final_assignment)
        # There should be at least one backtrack in a non-trivial UNSAT proof
        backtracks = [s for s in trace.steps if s.decision_type == DecisionType.BACKTRACK]
        self.assertGreaterEqual(len(backtracks), 1)

    def test_scheduling_sat(self):
        # 3 jobs, 2 slots, only one conflict -> SAT
        gen = SATGenerator(seed=1)
        inst = gen.generate_scheduling_problem(
            num_jobs=3,
            num_time_slots=2,
            conflicts=[(0, 1)],
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertTrue(result)
        self.assertIsNotNone(trace.final_assignment)
        self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))

    def test_scheduling_unsat(self):
        # 2 jobs, 1 slot, conflict between jobs -> UNSAT (both cannot be in the only slot)
        gen = SATGenerator(seed=2)
        inst = gen.generate_scheduling_problem(
            num_jobs=2,
            num_time_slots=1,
            conflicts=[(0, 1)],
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertFalse(result)
        self.assertIsNone(trace.final_assignment)

    def test_random_3sat_sanity(self):
        # Deterministic small random instance; verify internal consistency
        gen = SATGenerator(seed=42)
        inst = gen.generate_random_3sat(num_variables=6, num_clauses=18)
        solver = DPLLSolver(inst)
        sat = solver.solve()
        trace = solver.get_trace()

        self.assertIsNotNone(trace.final_result)
        if sat:
            self.assertIsNotNone(trace.final_assignment)
            self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))
        else:
            self.assertIsNone(trace.final_assignment)

if __name__ == "__main__":
    unittest.main()


