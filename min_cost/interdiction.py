
import pandas as pd
import pyomo
import pyomo.opt
import pyomo.environ as pe
import logging


class MinCostInterdiction:
    """
    An LP to compute single- or multi-commodity flow interdictions.
    """

    def __init__(self,
                 nodes: pd.DataFrame = None,
                 node_commodities: pd.DataFrame = None,
                 arcs: pd.DataFrame = None,
                 arc_commodities: pd.DataFrame = None,
                 attacks: int = 0,
                 arc_costs: str = "ArcCost",
                 interdiction_costs: str = "InterdictionCost",
                 solver: str = "glpk",
                 options_string: str = "",
                 keep_files = False):
        """
        All the node and arc object are dataframes with columns described below.

        nodes:
            ID

            Every node must appear as a line in the nodes dataframe.  You can 
            have additional columns as well.

        node_commodities:
            Node,Commodity,Demand

            Every commodity node imbalance that is not zero must appear in 
            node_commodities.

        arcs:
            StartNode,EndNode,Capacity,Attackable

            Every arc must appear in the arcs dataframe.  Also the arcs total 
            capacity and whether we can attack this arc ({0,1})

        arc_commodities:
            StartNode,EndNode,Commodity,Capacity,Cost,InterdictionCost

            This dataframe specifies the costs and capacities of moving each 
            commodity across each arc. If an (node, node, commodity) tuple 
            does not appear in this dataframe, then it means the commodity
            cannot flow across that edge.

            Enter -1 in the InterdictionCost column to signify BigM (set at 1e9)

            Additional columns can be added for different costing scenarios

        attacks:
            0 (default)

            The maximum number of attacks to employ against arcs in the network.
            Attacks are considered directional.

        arc_costs:
            "Cost" (default)

            The name of the column in the arc_commodities dataframe that 
            contains the arc cost per commodity. Useful for running additional scenarios.

        interdiction_costs:
            "InterdictionCost" (default)

            The name of the column in the arc_commodities dataframe that 
            contains the additional arc cost per commodity if that arc is 
            interdicted. Useful for running additional scenarios.

        solver:
            "cbc" (default)
            
            The name, as a string, of the solver to use.

        options_string:
            "" (default)

            Options to pass to the solver, as a single string.

        keep_files:
            Whether to keep the generated files for the solve and from the solver.

        Available outputs:
            interdictions: dataframe of chosen arcs
            flows: dataframe of arc flows by commodity
            total_costs: dataframe of objective values by commodity
            unsatisfied_commodities: dataframe of leftover supply and demand
            primal_solutions
            dual_solutions

        """

        # all properties and setters defined below __init__ routine

        # object properties
        self._nCmax = 1e9  # Largest possible value
        self._nodes = None  # list of nodes
        self._arcs = None  # list of arcs
        self._arc_commodities = None  # arc commodity combinations
        self._node_commodities = None  # supply/demands for commodities at nodes
        self._node_set = None  # node set formulation
        self._arc_set = None  # arc set formulation
        self._commodity_set = None  # commodity set formulation
        self._attacks = None  # the maximum number of interdictions available to use
        self._arc_costs = None  # name of the column detailing arc costs per commodity
        self._interdiction_costs = None  # name of the column containing interdiction costs per commodity
        self._solver = None  # name of the solver to use
        self._options_string = None  # string of options to pass to the solver
        self._primal = None  # the primal formulation (min cost flow)
        self._dual = None  # the dual formulation (interdictions)
        self._total_attacks = 0  # tally of total interdictions used

        self.keep_files = keep_files
        self.primal_solutions = None  # dataframe of solutions to the primal problem
        self.dual_solutions = None  # dataframe of solutions to the dual problem

        # set the interdiction and cost settings
        self.attacks = attacks
        self.arc_costs = arc_costs
        self.interdiction_costs = interdiction_costs

        # read network layout
        self.nodes = nodes
        self.arcs = arcs

        # load flow layout
        self.node_commodities = node_commodities
        self.arc_commodities = arc_commodities

        # set up solver params
        self.solver = solver
        self.options_string = options_string

        self.clear_results()

    def clear_results(self):
        """
        Populate blank results data.frames
        """
        # dataframes for saving results
        self.interdictions = pd.DataFrame()
        self.flows = pd.DataFrame()
        self.total_costs = pd.DataFrame()
        self.unsatisfied_commodities = pd.DataFrame()
        self.objective_values = pd.DataFrame()

    def formulate(self):
        """
        Formulate the primal and interdiction dual problems.
        """
        # create the two formulations
        self.create_primal()
        self.create_interdiction_dual()

    # NODES
    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, data):
        self._nodes = data.copy(deep=True)
        self._nodes.set_index(["ID"], inplace=True)
        self._nodes.sort_index(inplace=True)
        self._node_set = self._nodes.index.unique()

    # ARCS
    @property
    def arcs(self):
        return self._arcs

    @arcs.setter
    def arcs(self, data):
        self._arcs = data.copy(deep=True)
        self._arcs.set_index(["StartNode", "EndNode"], inplace=True)
        self._arcs.sort_index(inplace=True)
        self._arc_set = self.arcs.index.unique()

    # ARC COMMODITIES
    @property
    def arc_commodities(self):
        return self._arc_commodities

    @arc_commodities.setter
    def arc_commodities(self, data):
        self._arc_commodities = data.copy(deep=True)

        # for any column with -1 for the interdiction cost, set to nCmax (BigM)
        ic = self.interdiction_costs
        self._arc_commodities.loc[self._arc_commodities[ic] == -1, ic] = self._nCmax
        self._arc_commodities["xbar"] = 0
        self._arc_commodities.set_index(["StartNode",
                                         "EndNode", "Commodity"], inplace=True)
        self._arc_commodities.sort_index(inplace=True)

    # NODE COMMODITIES
    @property
    def node_commodities(self):
        return self._node_commodities

    @node_commodities.setter
    def node_commodities(self, data):
        self._node_commodities = data.copy(deep=True)
        self._node_commodities.set_index(["Node", "Commodity"], inplace=True)
        self._node_commodities.sort_index(inplace=True)
        self._commodity_set = self.node_commodities.index.levels[1].unique()

    # Attacks
    @property
    def attacks(self):
        return self._attacks

    @attacks.setter
    def attacks(self, number):
        self._attacks = number

    # Arc Costs
    @property
    def arc_costs(self):
        return self._arc_costs

    @arc_costs.setter
    def arc_costs(self, colname):
        self._arc_costs = colname

    # Interdiction costs
    @property
    def interdiction_costs(self):
        return self._interdiction_costs

    @interdiction_costs.setter
    def interdiction_costs(self, colname):
        self._interdiction_costs = colname

    # Solver
    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    # Options string
    @property
    def options_string(self):
        return self._options_string

    @options_string.setter
    def options_string(self, options):
        self._options_string = options

    def create_primal(self):
        """
        Formulate the primal model.

        The primal formulation computes minimum cost flows after interdiction.
        The interdiction is stored in arc_commodities[xbar].
        """

        model = pe.ConcreteModel()

        # Tell pyomo to read in dual-variable information from the solver
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        # Add the sets
        model._node_set = pe.Set(initialize=self._node_set)
        model._edge_set = pe.Set(initialize=self._arc_set, dimen=2)
        model._commodity_set = pe.Set(initialize=self._commodity_set)

        # Create the variables
        model.y = pe.Var(model._edge_set * model._commodity_set,
                         domain=pe.NonNegativeReals)
        model.UnsatSupply = pe.Var(model._node_set * model._commodity_set,
                                   domain=pe.NonNegativeReals)
        model.UnsatDemand = pe.Var(model._node_set * model._commodity_set,
                                   domain=pe.NonNegativeReals)

        # Create the objective
        # MINIMIZE...
        def obj_rule(_model):
            return sum((data[self.arc_costs] + data["xbar"] * data[self.interdiction_costs]) * _model.y[e]
                       for e, data in self.arc_commodities.iterrows()) + \
                   sum(self._nCmax * (_model.UnsatSupply[n] + _model.UnsatDemand[n])
                       for n, data in self._node_commodities.iterrows())

        model.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        # Create the constraints, one for each node
        # SUBJECT TO...
        def flow_bal_rule(_model, n, k):
            tmp = self.arcs.reset_index()
            predecessors = tmp.ix[tmp.EndNode == n, "StartNode"].values
            successors = tmp.ix[tmp.StartNode == n, "EndNode"].values
            lhs = sum(_model.y[(i, n, k)] for i in predecessors) - sum(_model.y[(n, i, k)] for i in successors)

            imbalance = self._node_commodities["Demand"].get((n, k), 0)
            supply_node = int(imbalance < 0)
            demand_node = int(imbalance > 0)
            rhs = (imbalance + _model.UnsatSupply[n, k] * supply_node -
                   _model.UnsatDemand[n, k] * demand_node)
            constr = (lhs == rhs)

            if isinstance(constr, bool):
                return pe.Constraint.Skip
            return constr

        model.FlowBalance = pe.Constraint(model._node_set * model._commodity_set, rule=flow_bal_rule)

        # Capacity constraints, one for each edge and commodity
        def capacity_rule(_model, i, j, k):
            capacity = self.arc_commodities["Capacity"].get((i, j, k), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return _model.y[(i, j, k)] <= capacity

        model.Capacity = pe.Constraint(model._edge_set *
                                       model._commodity_set, rule=capacity_rule)

        # Joint capacity constraints, one for each edge
        def joint_capacity_rule(_model, i, j):
            capacity = self.arcs["Capacity"].get((i, j), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return sum(_model.y[(i, j, k)]
                       for k in self._commodity_set) <= capacity

        model.JointCapacity = pe.Constraint(model._edge_set,
                                            rule=joint_capacity_rule)

        # Store the model
        self._primal = model

    def create_interdiction_dual(self):
        """
        Create the interdiction dual formulation.

        The interdiction dual taks the minimum cost primal formalation and
        transforms into the dual, maximum the objective value by targeting the 
        most critical arcs and subtracting the interdiction cost

        The interdiction dual is solved first to determine the interdicted
        arcs to feed into the primal for the minimum cost flow.
        """

        # Create the model
        model = pe.ConcreteModel()

        # Add the sets
        model._node_set = pe.Set(initialize=self._node_set)
        model._edge_set = pe.Set(initialize=self._arc_set, dimen=2)
        model._commodity_set = pe.Set(initialize=self._commodity_set)

        # Create the variables
        # the combination of all commodities and nodes
        model.rho = pe.Var(model._node_set * model._commodity_set,
                           domain=pe.Reals)
        # single and joint capacity constraints
        model.pi_single = pe.Var(model._edge_set * model._commodity_set,
                                domain=pe.NonPositiveReals)
        model.pi_joint = pe.Var(model._edge_set, domain=pe.NonPositiveReals)
        # whether we use the arc of not
        model.x = pe.Var(model._edge_set, domain=pe.Binary)

        # Create the objective
        # MAXIMIZE...
        def objective_function(_model):
            return sum(data["Capacity"] * _model.pi_joint[e]
                       for e, data in self.arcs.iterrows() if data["Capacity"] >= 0) + \
                   sum(data["Capacity"] * _model.pi_single[e]
                       for e, data in self.arc_commodities.iterrows() if data["Capacity"] >= 0) + \
                   sum(data["Demand"] * _model.rho[n]
                       for n, data in self._node_commodities.iterrows())

        model.OBJ = pe.Objective(rule=objective_function, sense=pe.maximize)

        # Create the constraints for y_ijk
        # SUBJECT TO...
        def edge_cost_constraint(_model, i, j, k):
            if (i, j, k) not in self.arc_commodities.index:
                return pe.Constraint.Skip

            attackable = int(self.arcs["Attackable"].get((i, j), 0))
            has_single_cap = int(self.arc_commodities["Capacity"].get((i, j, k),-1) >= 0)
            has_joint_cap = int(self.arcs["Capacity"].get((i, j), -1) >= 0)

            # edge costs constrained to by capacities, subject to flow, lessened by usage and interdictions,
            # totaling less than total cost
            return (_model.rho[(j, k)] - _model.rho[(i, k)] + _model.pi_single[(i, j, k)] * has_single_cap +
                    _model.pi_joint[(i, j)] * has_joint_cap -
                    (attackable * self.arc_commodities[self.interdiction_costs].get((i, j, k)) * _model.x[(i, j)])) <= \
                    self.arc_commodities[self.arc_costs].get((i, j, k))

        model.DualEdgeConstraint = pe.Constraint(model._edge_set * model._commodity_set, rule=edge_cost_constraint)

        # Create constraints for the UnsatDemand variables
        def unsat_constraint_rule(_model, n, k):
            if (n, k) not in self.node_commodities.index:
                return pe.Constraint.Skip
            imbalance = self.node_commodities["Demand"].get((n, k), 0)
            supply_node = int(imbalance < 0)
            demand_node = int(imbalance > 0)
            if supply_node:
                return -_model.rho[(n, k)] <= self._nCmax
            if demand_node:
                return _model.rho[(n, k)] <= self._nCmax
            return pe.Constraint.Skip

        model.UnsatConstraint = pe.Constraint(model._node_set * model._commodity_set, rule=unsat_constraint_rule)

        # Create the interdiction budget constraint 
        def attack_limit_rule(_model):
            _model.attacks = self.attacks
            return pe.summation(_model.x) <= _model.attacks

        model.AttackLimit = pe.Constraint(rule=attack_limit_rule)

        # Create, save the model
        self._dual = model

    def solve(self, formulate=True, tee=False):

        # if we haven't formulated yet, complete setup
        if formulate:
            self.formulate()

        # grab the solver and options from this
        solv = self.solver
        opts = self.options_string

        # reset the number of attacks used
        self._total_attacks = 0
        solver = pyomo.opt.SolverFactory(solv)

        # we redo the attacks every time we need to run
        # flow... make structure, set data to none, reconstruct and fill with 0s (preprocess)
        self._dual.AttackLimit.construct()
        self._dual.AttackLimit._constructed = False
        del self._dual.AttackLimit._data[None]
        self._dual.AttackLimit.reconstruct()
        self._dual.preprocess()

        # SOLVE the dual
        results = solver.solve(self._dual, tee=tee,
                               keepfiles=self.keep_files, options_string=opts)

        # Check that we actually computed an optimal solution, load results
        if results.solver.status != pyomo.opt.SolverStatus.ok:
            logging.warning("Check solver not ok?")

        if (results.solver.termination_condition !=
                pyomo.opt.TerminationCondition.optimal):
            logging.warning("Check solver optimality?")

        # grab solutions for dual and load
        self._dual.solutions.load_from(results)

        # Now put interdictions into xbar and solve primal
        for e in self.arcs.index:
            self.arc_commodities.ix[e, "xbar"] = self._dual.x[e].value

        # complete construction of the problem
        # the primal only needs the objective function redone, since the attacks play here
        self._primal.OBJ.construct()
        self._primal.OBJ._constructed = False
        self._primal.OBJ._init_sense = pe.minimize
        del self._primal.OBJ._data[None]
        self._primal.OBJ.reconstruct()
        self._primal.preprocess()

        # SOLVE the primal
        results = solver.solve(self._primal, tee=tee,
                               keepfiles=False, options_string=opts)

        # Check that we actually computed an optimal solution, load results
        if results.solver.status != pyomo.opt.SolverStatus.ok:
            logging.warning("Check solver not ok?")

        if (results.solver.termination_condition !=
                pyomo.opt.TerminationCondition.optimal):
            logging.warning("Check solver optimality?")

        self._primal.solutions.load_from(results)

        # populate the count of attacks... makes this useful for performing 
        # in a loop
        edges = sorted(self._arc_set)
        for e in edges:
            val = self._dual.x[e].value
            if val is not None and val > 0:
                self._total_attacks = self._total_attacks + 1

        self.save_solution()

    def echo(self):
        # grab solutions for primal and dual and save to object
        self.primal_solutions = self._primal.solutions
        self.dual_solutions = self._dual.solutions

        print()
        print("----------")
        print("Total cost with %d attack(s) = %.2f (primal) %.2f (dual)" % (
            self.attacks, self._primal.OBJ(), self._dual.OBJ()))
        print("Number of attack(s) implemented: %d" % self._total_attacks)

        # grab interdcitions
        for a in self._arc_set:
            if self._dual.x[a].value > 0:
                print("Interdict arc %s -> %s" % (str(a[0]), str(a[1])))
        print()

        # flows next
        for e0, e1 in self._arc_set:
            for k in self._commodity_set:
                flow = self._primal.y[(e0, e1, k)].value
                if flow > 0:
                    print("Flow on arc %s -> %s: %.2f %s" % (str(e0),
                                                             str(e1), flow, str(k)))
        print()

        # now echo any remaining supplies on various nodes. This happens
        # if the total cost to traverse exceeds _nCmax

        node_commodity_data = sorted(self.node_commodities.index)
        for n in node_commodity_data:
            remaining_supply = self._primal.UnsatSupply[n].value
            if remaining_supply > 0:
                print("Remaining supply of %s on node %s: %.2f" % (str(n[1]),
                                                                   str(n[0]), remaining_supply))

        for n in node_commodity_data:
            remaining_demand = self._primal.UnsatDemand[n].value
            if remaining_demand > 0:
                print("Remaining demand of %s on node %s: %.2f" % (str(n[1]),
                                                                   str(n[0]), remaining_demand))
        print()

    def save_solution(self):
        # grab solutions for primal and dual and save to object
        self.primal_solutions = self._primal.solutions
        self.dual_solutions = self._dual.solutions

        # save the objective values
        objective_vals = pd.DataFrame({"ArcCosts": self.arc_costs,
                            "InterdictionCosts": self.interdiction_costs,
                            "NumAttacks": self.attacks,
                            "Primal": self._primal.OBJ(),
                            "Dual": self._dual.OBJ()}, index=[self.attacks])
        self.objective_values = pd.concat([self.objective_values, objective_vals])


        # grab interdictions
        for a in self._arc_set:
            if self._dual.x[a].value > 0:
                idt = pd.DataFrame({"ArcCosts": self.arc_costs,
                                    "InterdictionCosts": self.interdiction_costs,
                                    "NumAttacks": self.attacks,
                                    "StartNode": str(a[0]),
                                    "EndNode": str(a[1])}, index=[self.attacks])
                self.interdictions = pd.concat([self.interdictions, idt])

        # flows next... save separate first to use with total costs
        flows = pd.DataFrame()
        for e0, e1 in self._arc_set:
            for k in self._commodity_set:
                flow = self._primal.y[(e0, e1, k)].value
                if flow > 0:
                    flw = pd.DataFrame({"ArcCosts": self.arc_costs,
                                        "InterdictionCosts": self.interdiction_costs,
                                        "NumAttacks": self.attacks,
                                        "StartNode": str(e0),
                                        "EndNode": str(e1),
                                        "Commodity": str(k),
                                        "Units": flow}, 
                                        index=[self.arc_costs])
                    flows = pd.concat([flows, flw])

        # control for if there are no flows
        if len(flows.index) != 0:
            # order the flows by linking start to end to next start
            starts = self.node_commodities.reset_index()
            starts = starts[starts["Demand"] < 0]
            ordered_flows = pd.DataFrame()
            for idx,data in starts.iterrows():
                node, commodity = data[['Node','Commodity']]
                flow = flows[(flows['StartNode'] == node) & (flows['Commodity'] == commodity)]
                while not flow.empty:
                    ordered_flows = pd.concat([ordered_flows, flow])
                    node = flow['EndNode'].tolist()
                    flow = flows[(flows['StartNode'].isin(node)) & (flows['Commodity'] == commodity)]

            # add flows to the accumulated data
            self.flows = pd.concat([self.flows, ordered_flows])

            # total flow costs (objective values by commodity)
            total_costs = flows
            joins = ["StartNode", "EndNode", "Commodity"]
            total_costs = pd.merge(total_costs, self.arc_commodities, 
                    left_on=joins, right_on=joins)
            total_costs["TotalCost"] = (total_costs["ArcCost"] + 
                    total_costs["xbar"] * 
                    total_costs["InterdictionCost"]) * total_costs["Units"]
            total_costs["ArcCosts"] = self.arc_costs
            total_costs["InterdictionCosts"] = self.interdiction_costs
            total_costs["NumAttacks"] = self.attacks
            total_costs = total_costs.groupby(by=["ArcCosts", "InterdictionCosts", "NumAttacks", "Commodity"]).sum()
            self.total_costs = pd.concat([self.total_costs, 
                total_costs[["TotalCost"]]])

        # unsatisfied flows last
        node_commodity_data = sorted(self.node_commodities.index)
        for n in node_commodity_data:
            remaining_supply = self._primal.UnsatSupply[n].value
            if remaining_supply > 0:
                supp = pd.DataFrame({"ArcCosts": self.arc_costs,
                                    "InterdictionCosts": self.interdiction_costs,
                                    "NumAttacks": self.attacks,
                                    "Type": "Supply",
                                    "Node": str(n[1]),
                                    "Commodity": str(n[0]),
                                    "Units": remaining_supply}, index=[self.attacks])
                self.unsatisfied_commodities = pd.concat([self.unsatisfied_commodities, supp])

        for n in node_commodity_data:
            remaining_demand = self._primal.UnsatDemand[n].value
            if remaining_demand > 0:
                dmd = pd.DataFrame({"ArcCosts": self.arc_costs,
                                     "InterdictionCosts": self.interdiction_costs,
                                     "NumAttacks": self.attacks,
                                     "Type": "Demand",
                                     "Node": str(n[1]),
                                     "Commodity": str(n[0]),
                                     "Units": remaining_demand}, index=[self.attacks])
                self.unsatisfied_commodities = pd.concat([self.unsatisfied_commodities, dmd])

# test routine
if __name__ == "__main__":

    # read in data and set parameters
    print("Reading in data...")
    node_data = pd.read_csv("../sample_nodes_data.csv")
    node_commodity_data = pd.read_csv("../sample_nodes_commodity_data.csv")
    arc_data = pd.read_csv("../sample_arcs_data.csv")
    arc_commodity_data = pd.read_csv("../sample_arcs_commodity_data.csv")
    arc_cost = "ArcCost"
    interdiction_cost = "InterdictionCost"

    # setup the object
    print("Creating LP...")
    m = MinCostInterdiction(nodes=node_data,
                                   node_commodities=node_commodity_data,
                                   arcs=arc_data,
                                   arc_commodities=arc_commodity_data,
                                   attacks=0,
                                   arc_costs=arc_cost,
                                   interdiction_costs=interdiction_cost)

    # add any desired heuristics here and run
    print("Solving LP...")
    maxAttacks = 5
    for i in range(maxAttacks+1):
        m.attacks = i
        m.solve()
        print(m.flows)
        print(m.unsatisfied_commodities)

    # display the solver output for the last run
    m._primal.display()
    m._dual.display()
    print("\n")
    print(m.interdictions)
    print(m.arc_commodities)
    print(m.objective_values)
    print(m.total_costs)
    print(m.flows)

