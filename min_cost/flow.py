
import pandas as pd
import pyomo
import pyomo.opt
import pyomo.environ as pe
import logging


class MinCostFlow:
    """An LP to compute single- or multi-commodity min cost flow (MCF)."""

    def __init__(self,
                 nodes: pd.DataFrame = None,
                 node_commodities: pd.DataFrame = None,
                 arcs: pd.DataFrame = None,
                 arc_commodities: pd.DataFrame = None,
                 arc_costs: str = "ArcCost",
                 solver: str = "cbc",
                 options_string: str = "mingap=0",
                 keep_files = False):
        """
        All the node and arc object are dataframes with columns described below.

        nodes:
            Node

            Every node must appear as a line in the nodes dataframe.  You can 
            have additional columns as well.

        node_commodities:
            Node,Commodity,SupplyDemand

            Every commodity node imbalance that is not zero must appear in 
            node_commodities.

        arcs:
            StartNode,EndNode,Capacity

            Every arc must appear in the arcs dataframe.

        arc_commodities:
            StartNode,EndNode,Commodity,Capacity,Cost

            This dataframe specifies the costs and capacities of moving each 
            commodity across each arc. If an (node, node, commodity) tuple 
            does not appear in this dataframe, then it means the commodity
            cannot flow across that edge.

        arc_costs:
            "Cost" (default)

            The name of the column in the arc_commodities dataframe that 
            contains the arc cost per commodity. Useful for running additional 
            scenarios.

        solver:
            "cbc" (default)
            
            The name, as a string, of the solver to use.

        options_string:
            "mingap=0" (default)

            Options to pass to the solver, as a single string.

        keep_files:
            Whether to keep the generated files for the solve and from the solver.

        Available outputs:
            flows: dataframe of arc flows by commodity
            primal_solutions

        """

        # all properties and setters defined below __init__ routine

        # object properties
        self._nCmax = 1e9
        self._nodes = None  # list of nodes
        self._arcs = None  # list of arcs
        self._arc_commodities = None  # arc commodity combinations
        self._node_commodities = None  # supply/demands for commodities at nodes
        self._node_set = None  # node set formulation
        self._arc_set = None  # arc set formulation
        self._commodity_set = None  # commodity set formulation
        self._arc_costs = None  # name of the column detailing arc costs per commodity
        self._solver = None  # name of the solver to use
        self._options_string = None  # string of options to pass to the solver
        self._primal = None  # the primal formulation (min cost flow)

        self.keep_files = keep_files
        self.primal_solutions = None  # dataframe of solutions to the primal problem

        # dataframes for saving results
        self.flows = pd.DataFrame()
        self.objective_values = pd.DataFrame()

        # set the interdiction and cost settings
        self.arc_costs = arc_costs

        # read network layout
        self.nodes = nodes
        self.arcs = arcs

        # load flow layout
        self.node_commodities = node_commodities
        self.arc_commodities = arc_commodities

        # set up solver params
        self.solver = solver
        self.options_string = options_string

    def formulate(self):
        """
        Formulate the primal and interdiction dual problems.
        """

        # create the two formulations
        self.create_primal()

    # NODES
    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, data):
        self._nodes = data.copy(deep=True)
        self._nodes.set_index(['Node'], inplace=True)
        self._nodes.sort_index(inplace=True)
        self._node_set = self._nodes.index.unique()

    # ARCS
    @property
    def arcs(self):
        return self._arcs

    @arcs.setter
    def arcs(self, data):
        self._arcs = data.copy(deep=True)
        self._arcs.set_index(['StartNode', 'EndNode'], inplace=True)
        self._arcs.sort_index(inplace=True)
        self._arc_set = self.arcs.index.unique()

    # ARC COMMODITIES
    @property
    def arc_commodities(self):
        return self._arc_commodities

    @arc_commodities.setter
    def arc_commodities(self, data):
        self._arc_commodities = data.copy(deep=True)
        self._arc_commodities.set_index(['StartNode',
                                         'EndNode', 'Commodity'], inplace=True)
        self._arc_commodities.sort_index(inplace=True)

    # NODE COMMODITIES
    @property
    def node_commodities(self):
        return self._node_commodities

    @node_commodities.setter
    def node_commodities(self, data):
        self._node_commodities = data.copy(deep=True)
        self._node_commodities.set_index(['Node', 'Commodity'], inplace=True)
        self._node_commodities.sort_index(inplace=True)
        self._commodity_set = self.node_commodities.index.levels[1].unique()

    # Arc Costs
    @property
    def arc_costs(self):
        return self._arc_costs

    @arc_costs.setter
    def arc_costs(self, colname):
        self._arc_costs = colname

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
            return sum(data[self.arc_costs]  * _model.y[e]
                       for e, data in self.arc_commodities.iterrows()) + \
                   sum(self._nCmax * (_model.UnsatSupply[n] + _model.UnsatDemand[n])
                       for n, data in self._node_commodities.iterrows())

        model.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        # Create the constraints, one for each node
        # SUBJECT TO...
        def flow_bal_rule(_model, n, k):
            tmp = self.arcs.reset_index()
            successors = tmp.ix[tmp.StartNode == n, 'EndNode'].values
            predecessors = tmp.ix[tmp.EndNode == n, 'StartNode'].values
            lhs = sum(_model.y[(i, n, k)] for i in predecessors) - sum(_model.y[(n, i, k)] for i in successors)

            imbalance = self._node_commodities['SupplyDemand'].get((n, k), 0)
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
            capacity = self.arc_commodities['Capacity'].get((i, j, k), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return _model.y[(i, j, k)] <= capacity

        model.Capacity = pe.Constraint(model._edge_set *
                                       model._commodity_set, rule=capacity_rule)

        # Joint capacity constraints, one for each edge
        def joint_capacity_rule(_model, i, j):
            capacity = self.arcs['Capacity'].get((i, j), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return sum(_model.y[(i, j, k)]
                       for k in self._commodity_set) <= capacity

        model.JointCapacity = pe.Constraint(model._edge_set,
                                            rule=joint_capacity_rule)

        # Store the model
        self._primal = model

    def solve(self, tee=False):

        # complete setup
        self.formulate()

        # grab the solver and options from this
        solver = pyomo.opt.SolverFactory(self.solver)

        # complete construction of the problem
        self._primal.OBJ.construct()
        self._primal.OBJ._constructed = False
        self._primal.OBJ._init_sense = pe.minimize
        del self._primal.OBJ._data[None]
        self._primal.OBJ.reconstruct()
        self._primal.preprocess()

        # SOLVE the primal

        results = solver.solve(self._primal, tee=tee,
                               keepfiles=self.keep_files, options_string=self.options_string)

        # Check that we actually computed an optimal solution, load results
        if results.solver.status != pyomo.opt.SolverStatus.ok:
            logging.warning('Check solver not ok?')

        if (results.solver.termination_condition !=
                pyomo.opt.TerminationCondition.optimal):
            logging.warning('Check solver optimality?')

        self._primal.solutions.load_from(results)

        self.save_solution()

    def echo(self):
        # grab solutions for primal and dual and save to object
        self.primal_solutions = self._primal.solutions

        print()
        print('----------')

        # flows
        for e0, e1 in self._arc_set:
            for k in self._commodity_set:
                flow = self._primal.y[(e0, e1, k)].value
                if flow > 0:
                    print('Flow on arc %s -> %s: %.2f %s' % (str(e0),
                                                             str(e1), flow, str(k)))
        print()

        # now echo any remaining supplies on various nodes. This happens

        node_commodity_data = sorted(self.node_commodities.index)
        for n in node_commodity_data:
            remaining_supply = self._primal.UnsatSupply[n].value
            if remaining_supply > 0:
                print('Remaining supply of %s on node %s: %.2f' % (str(n[1]),
                                                                   str(n[0]), remaining_supply))

        for n in node_commodity_data:
            remaining_demand = self._primal.UnsatDemand[n].value
            if remaining_demand > 0:
                print('Remaining demand of %s on node %s: %.2f' % (str(n[1]),
                                                                   str(n[0]), remaining_demand))
        print()

    def save_solution(self):
        # grab solutions for primal and dual and save to object
        self.primal_solutions = self._primal.solutions

        # save the objective values
        objective_vals = pd.DataFrame({"ArcCosts": self.arc_costs,
                            "Primal": self._primal.OBJ()},
                            index=[self.arc_costs])
        self.objective_values = pd.concat([self.objective_values, objective_vals])


        # flows next
        for e0, e1 in self._arc_set:
            for k in self._commodity_set:
                flow = self._primal.y[(e0, e1, k)].value
                if flow > 0:
                    flw = pd.DataFrame({"ArcCosts": self.arc_costs,
                                        "From": str(e0),
                                        "To": str(e1),
                                        "Commodity": str(k),
                                        "Units": flow}, index=[self.arc_costs])
                    self.flows = pd.concat([self.flows, flw])

        # unsatisfied flows last
        node_commodity_data = sorted(self.node_commodities.index)
        for n in node_commodity_data:
            remaining_supply = self._primal.UnsatSupply[n].value
            if remaining_supply > 0:
                supp = pd.DataFrame({"ArcCosts": self.arc_costs,
                                    "Type": "Supply",
                                    "Node": str(n[1]),
                                    "Commodity": str(n[0]),
                                    "Units": remaining_supply}, index=[self.arc_costs])
                self.unsatisfied_commodities = pd.concat([self.unsatisfied_commodities, supp])

        for n in node_commodity_data:
            remaining_demand = self._primal.UnsatDemand[n].value
            if remaining_demand > 0:
                dmd = pd.DataFrame({"ArcCosts": self.arc_costs,
                                     "Type": "Demand",
                                     "Node": str(n[1]),
                                     "Commodity": str(n[0]),
                                     "Units": remaining_demand}, index=[self.arc_costs])
                
