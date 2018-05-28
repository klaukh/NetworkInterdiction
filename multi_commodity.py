# TODO: For interdiction costs, add option - like Capacity - where a -1 then translates to infinity (BigM?)
# TODO: Change around the Supply and Demand signs
# TODO: Add NEOS solver manager option

import pandas
import pyomo
import pyomo.opt
import pyomo.environ as pe
import logging
import csv

# NOTE: we won't consider Node interdiction, as Node interdiction can easily be represented by
# bifurcating each node into two and placing an unconstrained, no-cost arc between them

class MultiCommodityInterdiction:
    """A class to compute multicommodity flow interdictions."""

    def __init__(self, node_file, node_commodity_file, arc_file, 
            arc_commodity_file, attacks = 0,
            arc_cost = "ArcCost", 
            interdiction_cost = "InterdictionCost",
            outfile = "",
            solver = "cbc",
            options_string = "mingap=0"):
        """
        All the files are CSVs with columns described below.

        - node_file:
            Node

        Every node must appear as a line in the node_file.  You can have 
        additional columns as well.

        - node_commodity_file:
            Node,Commodity,SupplyDemand

        Every commodity node imbalance that is not zero must appear in the 
        node_commodity_file

        - arc_file:
            StartNode,EndNode,Capacity,Attackable

        Every arc must appear in the arc_file.  Also the arcs total capacity 
        and whether we can attack this arc.

        - arc_commodity_file:
            StartNode,EndNode,Commodity,Capacity,Cost,InterdictionCost

        This file specifies the costs and capacities of moving each commodity 
        across each arc. If an (node, node, commodity) tuple does not appear 
        in this file, then it means the commodity
        cannot flow across that edge.

        - attacks:

        The maximum number of attacks to employ against arcs in the network. 
        Attacks are considered directional.

        - arc_cost:

        The column reference in the arc_commodity_file for the base arc costs.

        - interdiction_cost:

        The column reference in the arc_commodity_file for the additional 
        arc cost imposed if an interdiction is placed there.

        - outfile:

        String to prepend to output files generated.

        - solver:
        The name, as a string, of the solver to use.

        - options_string:
        Options to pass to the solver, as a single string.

        """

        # Read in the node_data
        self.node_data = pandas.read_csv(node_file)
        self.node_data.set_index(['Node'], inplace=True)
        self.node_data.sort_index(inplace=True)

        # Read in the node_commodity_data
        self.node_commodity_data = pandas.read_csv(node_commodity_file)
        self.node_commodity_data.set_index(['Node', 'Commodity'], inplace=True)
        self.node_commodity_data.sort_index(inplace=True)

        # Read in the arc_data
        self.arc_data = pandas.read_csv(arc_file)
        self.arc_data.set_index(['StartNode', 'EndNode'], inplace=True)
        self.arc_data.sort_index(inplace=True)

        # Read in the arc_commodity_data
        self.arc_commodity_data = pandas.read_csv(arc_commodity_file)
        self.arc_commodity_data['xbar'] = 0
        self.arc_commodity_data.set_index(['StartNode', 
            'EndNode', 'Commodity'], inplace=True)
        self.arc_commodity_data.sort_index(inplace=True)
        # Can df.reset_index() to go back

        # set the interdiction and cost settings
        self.attacks = attacks
        self.arc_cost = arc_cost
        self.interdiction_cost = interdiction_cost

        self.node_set = self.node_data.index.unique()
        self.commodity_set = self.node_commodity_data.index.levels[1].unique()
        self.arc_set = self.arc_data.index.unique()

        # set up solver params
        self.solver = solver
        self.options_string = options_string

        # outfile arg
        self.outfile = outfile

        # Compute nCmax
        self.nCmax = len(self.node_set) * \
            (self.arc_commodity_data[self.arc_cost].max() + \
            self.arc_commodity_data[self.interdiction_cost].max())


        self.createPrimal()
        self.createInterdictionDual()

    def createPrimal(self):
        """Create the primal model.
        This is used to compute flows after interdiction.  The interdiction 
        is stored in arc_commodity_data.xbar."""

        model = pe.ConcreteModel()
        # Tell pyomo to read in dual-variable information from the solver
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        # Add the sets
        model.node_set = pe.Set(initialize=self.node_set)
        model.edge_set = pe.Set(initialize=self.arc_set, dimen=2)
        model.commodity_set = pe.Set(initialize=self.commodity_set)

        # Create the variables
        model.y = pe.Var(model.edge_set * model.commodity_set, domain=pe.NonNegativeReals)
        model.UnsatSupply = pe.Var(model.node_set * model.commodity_set, domain=pe.NonNegativeReals)
        model.UnsatDemand = pe.Var(model.node_set * model.commodity_set, domain=pe.NonNegativeReals)

        # Create the objective
        def obj_rule(model):
            return sum((data[self.arc_cost] + data['xbar'] * data[self.interdiction_cost]) * model.y[e] \
                       for e, data in self.arc_commodity_data.iterrows()) + \
                   sum(self.nCmax * (model.UnsatSupply[n] + model.UnsatDemand[n]) \
                       for n, data in self.node_commodity_data.iterrows())

        model.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        # Create the constraints, one for each node
        def flow_bal_rule(model, n, k):
            tmp = self.arc_data.reset_index()
            successors = tmp.ix[tmp.StartNode == n, 'EndNode'].values
            predecessors = tmp.ix[tmp.EndNode == n, 'StartNode'].values
            lhs = sum(model.y[(i, n, k)] for i in predecessors) - sum(model.y[(n, i, k)] for i in successors)
            imbalance = self.node_commodity_data['SupplyDemand'].get((n, k), 0)
            supply_node = int(imbalance < 0)
            demand_node = int(imbalance > 0)
            rhs = (imbalance + model.UnsatSupply[n, k] * (supply_node) - model.UnsatDemand[n, k] * (demand_node))
            constr = (lhs == rhs)
            if isinstance(constr, bool):
                return pe.Constraint.Skip
            return constr

        model.FlowBalance = pe.Constraint(model.node_set * model.commodity_set, rule=flow_bal_rule)

        # Capacity constraints, one for each edge and commodity
        def capacity_rule(model, i, j, k):
            capacity = self.arc_commodity_data['Capacity'].get((i, j, k), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return model.y[(i, j, k)] <= capacity

        model.Capacity = pe.Constraint(model.edge_set * model.commodity_set, rule=capacity_rule)

        # Joint capacity constraints, one for each edge
        def joint_capacity_rule(model, i, j):
            capacity = self.arc_data['Capacity'].get((i, j), -1)
            if capacity < 0:
                return pe.Constraint.Skip
            return sum(model.y[(i, j, k)] for k in self.commodity_set) <= capacity

        model.JointCapacity = pe.Constraint(model.edge_set, rule=joint_capacity_rule)

        # Store the model
        self.primal = model

    def createInterdictionDual(self):
        # Create the model
        model = pe.ConcreteModel()

        # Add the sets
        model.node_set = pe.Set(initialize=self.node_set)
        model.edge_set = pe.Set(initialize=self.arc_set, dimen=2)
        model.commodity_set = pe.Set(initialize=self.commodity_set)

        # Create the variables
        model.rho = pe.Var(model.node_set * model.commodity_set, domain=pe.Reals)
        model.piSingle = pe.Var(model.edge_set * model.commodity_set, domain=pe.NonPositiveReals)
        model.piJoint = pe.Var(model.edge_set, domain=pe.NonPositiveReals)

        model.x = pe.Var(model.edge_set, domain=pe.Binary)

        # Create the objective
        def obj_rule(model):
            return sum(
                data['Capacity'] * model.piJoint[e] for e, data in self.arc_data.iterrows() if data['Capacity'] >= 0) + \
                   sum(data['Capacity'] * model.piSingle[e] \
                       for e, data in self.arc_commodity_data.iterrows() if data['Capacity'] >= 0) + \
                   sum(data['SupplyDemand'] * model.rho[n] for n, data in self.node_commodity_data.iterrows())

        model.OBJ = pe.Objective(rule=obj_rule, sense=pe.maximize)

        # Create the constraints for y_ijk
        def edge_constraint_rule(model, i, j, k):
            if (i, j, k) not in self.arc_commodity_data.index:
                return pe.Constraint.Skip
            attackable = int(self.arc_data['Attackable'].get((i, j), 0))
            hasSingleCap = int(self.arc_commodity_data['Capacity'].get((i, j, k), -1) >= 0)
            hasJointCap = int(self.arc_data['Capacity'].get((i, j), -1) >= 0)
            return model.rho[(j, k)] - model.rho[(i, k)] + model.piSingle[(i, j, k)] * hasSingleCap + \
                model.piJoint[(i, j)] * hasJointCap - \
                (self.arc_commodity_data[self.interdiction_cost].get((i, j, k)) * model.x[(i, j)] * attackable) \
                <= self.arc_commodity_data[self.arc_cost].get((i, j, k))

        model.DualEdgeConstraint = pe.Constraint(model.edge_set * model.commodity_set, rule=edge_constraint_rule)

        # Create constraints for the UnsatDemand variables
        def unsat_constraint_rule(model, n, k):
            if (n, k) not in self.node_commodity_data.index:
                return pe.Constraint.Skip
            imbalance = self.node_commodity_data['SupplyDemand'].get((n, k), 0)
            supply_node = int(imbalance < 0)
            demand_node = int(imbalance > 0)
            if supply_node:
                return -model.rho[(n, k)] <= self.nCmax
            if demand_node:
                return model.rho[(n, k)] <= self.nCmax
            return pe.Constraint.Skip

        model.UnsatConstraint = pe.Constraint(model.node_set * model.commodity_set, rule=unsat_constraint_rule)

        # Create the interdiction budget constraint 
        def block_limit_rule(model):
            model.attacks = self.attacks
            return pe.summation(model.x) <= model.attacks

        model.BlockLimit = pe.Constraint(rule=block_limit_rule)

        # Create, save the model
        self.Idual = model

    def solve(self, tee=False):
        # grab the solver and options from this
        solv = self.solver
        opts = self.options_string

        # reset the number of attacks used
        self.total_attacks = 0
        solver = pyomo.opt.SolverFactory(solv)

        # Solve the dual first
        self.Idual.BlockLimit.construct()
        self.Idual.BlockLimit._constructed = False
        del self.Idual.BlockLimit._data[None]
        self.Idual.BlockLimit.reconstruct()
        self.Idual.preprocess()
        results = solver.solve(self.Idual, tee=tee, keepfiles=False, options_string=opts)

        # Check that we actually computed an optimal solution, load results
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):
            logging.warning('Check solver optimality?')

        self.Idual.solutions.load_from(results)
        # Now put interdictions into xbar and solve primal

        for e in self.arc_data.index:
            self.arc_commodity_data.ix[e, 'xbar'] = self.Idual.x[e].value

        self.primal.OBJ.construct()
        self.primal.OBJ._constructed = False
        self.primal.OBJ._init_sense = pe.minimize
        del self.primal.OBJ._data[None]
        self.primal.OBJ.reconstruct()
        self.primal.preprocess()
        results = solver.solve(self.primal, tee=tee, keepfiles=False, options_string="mipgap=0")

        # Check that we actually computed an optimal solution, load results
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):
            logging.warning('Check solver optimality?')

        self.primal.solutions.load_from(results)

        # populate the count of attacks... makes this useful for performing in a loop
        edges = sorted(self.arc_set)
        for e in edges:
            if self.Idual.x[e].value > 0:
                self.total_attacks = self.total_attacks + 1

    def printSolution(self):
        delim = "_"

        print()
        print('\nUsing %d attacks:' % self.attacks)
        print()
        edges = sorted(self.arc_set)
        csvfile = delim.join((self.outfile, self.arc_cost, 
                self.interdiction_cost, 
                "Interdictions.csv"))

        # write outputs to files and print to console
        # interdictions first
        with open(csvfile, "w+") as output:
          writer = csv.writer(output, lineterminator="\n")
          for e in edges:
              if self.Idual.x[e].value > 0:
                  writer.writerows([[self.arc_cost, self.interdiction_cost, 
                      self.attacks, str(e[0]), str(e[1])]])
                  print('Interdict arc %s -> %s' % (str(e[0]), str(e[1])))
        output.close()
        print()

        # flows next
        csvfile = delim.join((self.outfile, self.arc_cost, 
                self.interdiction_cost, 
                "Flows.csv"))
        with open(csvfile, "w+") as output:
            writer = csv.writer(output, lineterminator="\n")
            for e0, e1 in self.arc_set:
                for k in self.commodity_set:
                    flow = self.primal.y[(e0, e1, k)].value
                    if flow > 0:
                        print('Flow on arc %s -> %s: %.2f %s' % (str(e0), 
                            str(e1), flow, str(k)))
                        writer.writerows([[self.arc_cost, 
                            self.interdiction_cost, self.attacks,
                                          str(e0), str(e1), flow, str(k)]])

        output.close()

        # now echo any remaining supplies on various nodes. This happens
        # if the total cost to traverse exceeds nCmax
        print()
        print('----------')
        print('Total cost with %d attack(s) = %.2f (primal) %.2f (dual)' % (
        self.attacks, self.primal.OBJ(), self.Idual.OBJ()))
        print('Number of attack(s) implemented: %d' % (self.total_attacks))
        nodes = sorted(self.node_commodity_data.index)

        for n in nodes:
            remaining_supply = self.primal.UnsatSupply[n].value
            if remaining_supply > 0:
                print('Remaining supply of %s on node %s: %.2f' % (str(n[1]), 
                    str(n[0]), remaining_supply))

        for n in nodes:
            remaining_demand = self.primal.UnsatDemand[n].value
            if remaining_demand > 0:
                print('Remaining demand of %s on node %s: %.2f' % (str(n[1]), 
                    str(n[0]), remaining_demand))


