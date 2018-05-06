# based on the Pyomo Example Gallery

import pandas
import pyomo
import pyomo.opt
import pyomo.environ as pe
import logging

class MultiCommodityInterdiction:
   """A class to calculat multi-commotidy flow interdictions and 
   shortest paths, with multiple scenario options."""

   def __init(self, 
          node_file, 
          node_commodity_file, 
          arc_file, 
          arc_commodity_file,
          attacks = 0,
          interdiction_scenario = "Baseline",
          cost_scenario = "Baseline"):

       """
       All files should be in *.csv format with columns as below.

       - node_file:
           Node (the unique list of node names). Other columns can be 
           included but are ignored by this class.

       - node_commodity_file:
         Node,Commodity,SupplyDemand

         The tuples of node, commodity, and supply (positive integer) or 
         demand (negative integer). Nodes with zero SupplyDemand can be 
         omitted

       - arc_file:
         StartNode,EndNode,Capacity,Baseline

       Every arc must appear in the arcfile, with the start node, end node, 
       and the capacity. The Baseline column - and additional name 
       columns - define whether or not the arc can be interdicted. For 
       this class the column should be a binary entry. Baseline and any 
       additional interdiction scenario columns must be matched in the 
       arc_commodity_file

       - arc_commodity_file:
         StartNode,EndNode,Commodity,Capacity,Cost,Baseline

       This file specifies the costs and capacities of moving each commodity 
       across each arc.  If a (node, node, commodity) tuple does not 
       appear in this file, then the commodity cannot flow across that 
       edge.  The arc_commodity_file may include one or more columns for 
       the Cost (cost to each commodity for travel over that arc) and 
       interdiction costs (Baseline; the extra cost imposed on the commodity 
       for travel across that arc when the arc is interdicted.  The column 
       names above are the default column names, but others may be included 
       and set in the function call or direct setting the class member.

       """

       # node_file ingest
       self.node_data = pandas.read_csv(node_file)
       self.node_data.set_index(['Node'], inplace = True)
       self.node_data.sort_index(inplace = True)

       # node_commodity_data ingest
       self.node_commodity_data = pandas.read_csv(node_commodity_file)
       self.node_commodity_data.set_index['Node', 'Commodity'],
          inplace = True)
       self.node_commodity_data.sort_index(inplace = True)

       # arc_file ingest
       self.arc_data = pandas.read_csv(arc_file)
       self.arc_data.set_index(['StartNode', 'EndNode'], inplace = True)
       self.arc_data.sort_index(inplace = True)

       # arc_commodity_data ingest
       self.arc_commodity_data = pandas.read_csv(arc_commodity_file)
       self.arc_commodity_data.set_index(['StartNode','EndNode','Commodity'],
          inplace=True)
       self.arc_commodity_data.sort_index(inplace=True)
       # a special line to house whether we interdict this arc
       self.arc_commodity_data['xbar'] = 0 


       # interdiction, cost parameters. These can also be set externally
       self.attacks = attacks
       self.interdiction_scenario = interdiction_scenario
       self.cost_scenario = cost_scenario

       # additional internal data
       self.node_set = self.node_data.index.unique()
       self.commodity_set = self.node_commdity_data.index.levels[1].unique()
       self.arc_set = self.arc_data.index.unique()


       # Compute nCmax... this is the absolute maximum cost that can be
       # incurred... would happen in the scenario that a commodity had
       # to cross every arc and incur the max possible cost across all
       # commodities and interdiction penalties
       self.nCmax = len(self.node_set) * 
          (self.arc_commodity_data[self.cost_scenario].max() +
           self.arc_commodity_data[self.cost_scenario].max())

       # the primal model... for shortest path flow routing
       def createPrimal(self):
           """Create the primal... the shortest path flow routing. Calculated
           after the interdictions - stored in arc_commodity_data.xbar."""

           # use a concrete pyomo model
           model = pe.ConcreteModel()

           # read the dual's variable information from the solver
           model.dual = pe.Suffix(direction = pe.Suffix.IMPORT)

           # add model sets
           model.node_set = pe.Set(initialize = self.node_set)
           model.arc_set = pe.Set(initialize = self.arc_set, dimen = 2)
           model.commodity_set = pe.Set(initialize = self.commodity_set)

           # create variables
           # Unsat = unsatisfied
           model.y = pe.Var(model.arc_set * model.commodity_set, 
              domain = pe.NonNegativeReals)
           model.UnsatSupply = pe.Var(model.node_set * model.commodity_set,
              domain = pe.NonNegativeReals)
           model.UnsatDemand = pe.Var(model.node_set * model.commodity_set,
              domain = pe.NonNegativeReals)

           # create objective function
           def objective_function(model):
               # iterrate over the arcs, summing the costs incurred for
               # flow over those arcs, for each commodity, satisfying
               # the balance of flow constraints, and forcing the attempt
               # to flow resources to the end by create a harsh penalty
               # for unsatisfied demand or supply
               return sum( (data[self.cost_scenario] + data['xbar'] * 
                   data[self.interdiction_scenario] * model.y[e] \
                   for e,data in self.arc_commodity_data.iterrows()) +
                      sum(self.nCmax * (model.UnsatSupply[n] + 
                         model.UnsatDemand[n]) \
                         for n.data in self.node_commodity_data.iterrows())

           # assign the rule to the model... it's a minimize rule
           model.OBJ = pe.Objective(fule = objective_function, 
               sense = pe.minnimize)

           # balance of flow rules
           def flow_balance_rule(model, n, k): 
              tmp = self.arc_data.reset_index()
              # create indexed records of each incoming and outgoing arc for
              # a given node-commodity pairing
              predecessors = tmp.ix[tmp.EndNode == n, 'StartNode'].values
              successors = tmp.ix[tmp.StartNode == n, 'EndNode'].values
             
              # iterrate over each and create the total flow balance sum
              lhs = sum(model.y[(i,n,k)] for i in predecessors) - 
                 sum(model.y[(n,i,k)] for i in successors)

              # get the imbalance, with zero for missing values
              imbalance = self.node_commodity_dat['SupplyDemand'].get((n,k),0)
              supply_node = int(imbalance > 0)
              demand_node = int(imbalance < 0)

              # now create the RHS sum
              rhs = (imbalance + model.UnsatSupply[n,k] * (supply_node) - 
                  model.UnsatDemand[n,k] * (demand_node))

              # the constraint is just that they must be equal
              constr = (lhs==rhs)
              if isinstance(constr, bool): return pe.Constraint.Skip

              return constr
           # end of flow_balance_rule

           # capacity contraints
           def capacity_rule(model, i, j, k):
              # get the stated capacity, -1 if absent (and skip those)
              capacity = self.arc_commodity_data['Capacity']get((i,j,k,-1)
              if capacity < 0: return pe.Constraint.skip
            
              return model.y[(i,j,k)] <= capacity
           # end of arc capacity constraints

           # combined (joint) capacity for each edge
           def joint_capacity_rule(model, i, j):
              capacity = self.arc_data['Capacity'].get((i,j),-1)
              if capacity < 0: return pe.Constraint.Skip
              
              return sum(model.y[(i,j,k)] for k in self.commodity_set) <= 
                 capacity
           # end of combined arc capacity

           # create the flow balance rule for each node-commodity pair
           # the flow_balance_rule() skips those without such a pairing
           model.FlowBalance = pe.Constraint(model.node_set * 
               model.commodity_set, rule=flow_balance_rule)
           # create the individual capacity constraints for each 
           # arc-commodity pairing
           model.Capacity = pe.Constraint(model.arc_set * 
               model.commodity_set, rule = capacity_rule)
           # create the joint capacity for each arc
           model.JointCapacity = pe.constraint(model.arc_set, 
               rule = joint_capacity_rule)

           # save the primal
           self.primal = model


        # now create the dual
        def createInterdictionDual(self):
           # start a concrete model
           model = pe.concreteModel()

           # add the sets
           model.node_set = pe.Set(initialize = self.node_set)
           model.arc_set = pe.Set(initialize = self.arc_set, dimen = 2)
           model.commodity_set = pe.Set(initialize = self.commodity_set)

           # Create the variables
           model.rho = pe.Var(model.node_set * model.commodity_set, 
               domain = pe.Reals)
           model.piSingle = pe.Var(model.arc_set * model.commodity_set,  
               domain = pe.NonPositiveReals)
           model.piJoint = pe.Var(model.arc_set, domain = pe.NonPositiveReals)
           # this next one is the binary for whether we've interdicted the arc
           model.x = pe.Var(model.edge_set, domain=pe.Binary)

           # objective function for the dual
           def objection_function(model):
              return \
                 sum(data['Capacity'] * model.piJoint[e] \
                    for e,data in self.arc_data.iterrows() \
                       if data['Capacity'] >= 0 +
                    sum(data['Capacity'] * model.piSingle[e] \
                     for e,data in self.arc_commodity_data.iterrows() \
                        if data['Capacity'] >= 0) +
                        sum(data['SupplyDemand'] * model.rho[n] \
                           for n,data in self.node_commodity_data.iterrows())
                    


            # contraints for y_ijk
            def arc_constraint_rule(model, i, j, k):
               # skip if not in the commodity arc list
               if(i, j, k) not in self.arc_commodity_data.index:
                  return pe.Constraint.Skip
               attackable = 
                  int(self.arc_data[self.interdiction_scenario].get((i,j),0))

               hasSingleCap = 
                  int(self.arc_commodity_data['Capacity'].get((i,j,k),-1) >= 0

               hasJointCap = 
                  int(self.arc_data['Capacity'].get((i,j),-1) >= 0)

               return model.rho[(j,k)] - model.rho[(i,k)] + 
                  model.piSingle[(i,j,k)] * hasSingleCap + 
                  model.piJoint[(i,j)] * hasJointCap <= 
                  self.arc_commodity_data[self.cost_scenario].get((i,j,k)) +
                      (2 * self.nCmax + 1) * model.x[(i,j)] * attackable


            # constraints for unsatisfied demand variabes
            def unsat_constraint_rule(model, n, k):
               if (n,k) not in self.node_commodity_data.index:
                  return pe.Constraint.Skip
               
               imbalance = 
                  self.node_commodity_data['SupplyDemand'].get((n,k),0)
               supply_node = int(imbalance > 0)
               demand_node = int(imbalance < 0)

               if (supply_node):
                 return -model.rho[(n,k)] >= self.nCmax
               if (demand_node):
                 return model.rho[(n,k)] <= self.nCmax
               return pe.Constraint.Skip


            # create the interdiction budget constraint
            def interdiction_limit_rule(model):
               model.attacks = self.attacks
               return pe.summation(model.x) <= model.attacks

            # create the dual model
            model.OBJ = pe.Objective(rule = objective_function, 
                sense = pe.maximize)
            # subject to
            model.DualEdgeConstraint = pe.constraint(model.arc_set * 
                model.commodity_set, rule = arc_constraint_rule)
            model.UnsatConstraint = pe.Constraint(model.node_set * 
                model.commodity_set, rule = unsat_constraint_rule)
            model.AttackLimit = pe.Constraint(rule = inerdiction_limit_rule)

            # save the dual
            self.Idual = model

       # create the the two formulations
       self.createPrimal()
       self.createInterdictionDual()


        # solving the problem
        def solve(self, tee = False):
           # definine the solver to be used (put manager here as rqd)
           solver = pyomo.opt.SolverFactor('cbc')

           # counting the total attacks... provides a hook for early exit if
           # the model stops finding arcs to interdict
           self.total_attacks = 0
           
           ### solve the dual first ###
           # reset the model's interdiction tracker... repeat this if doing
           # some sort of move-countermove system
           self.Idual.AttackLimit.construct()
           self.Idual.AttackLimit._constructed = False
           del self.Idual.AttackLimit._data[None]
           self.Idual.AttackLimit.reconstruct()
           self.Idual.preprocess()
           results = solver.solve(self.Idual, tee = tee, keepfiles = False,
              options_string = 'mingap=0')

           # perform some checks
           if (results.solver.status != pyomo.opt.SolverStatus.ok):
               logging.warning('Solver error. Check solver.')

           if (results.solver.termination_condition != 
               pyomo.opt.TerminationCondition.optimal):
               logging.warning('Not optimal solution. Check solver settings')

           # save results, load interdictions into xbar and solve the primal 
           # for shortest path flow
           self.Idual.solutions.load_from(results)

           for e in self.arc_data.index:
               self.arc_commodity_data.ix[e, 'xbar'] = self.Idual.x[e].value

           ### solve the primal now ###
           # reset the model's formulation then solve
           self.primal.OBJ.construct()
           self.primarl.OBJ_constructed = False
           self.primal.OBJ._init_sense = pe.minimize
           del self.primal.OBJ._data[None]
           self.primal.OBJ.reconstruct()
           self.primal.preprocess()
           results = solver.solve(self.primal, tee = tee, keepfiles = False,
              options_string = 'mingap=0')

           # check for optimal solution, then load results
           if (results.solver.status != pyomo.opt.SolverStatus.ok):
               logging.warning('Solver error. Check solver.')
           if (results.solver.termination_condition != 
               pyomo.opt.TerminationCondition.optimal):
               logging.warning('Not optimal solution. Check solver settings')

           self.primal.solution.load_from(results)

           # populate the count of attacks... makes this useful for 
           # performing in a loop
           arcs = sorted(self.arc_set)
           for e in arcs:
              if self.Idual.x[e].value > 0:
                 self.total_attacks = self.total_attacks + 1

    def printSolution(self):
       # settings info
       print()
       print('\nUsing %d attacks:' %self.attacks)
       print()

       # print optimal interdictions
       arcs = sorted(self.arc_set)
       for e in arcs:
          if self.Idual.x[e].value > 0:
             print('Interdict arc %s -> %s' %(str(e[0]), str(e[1])))

       print()
       
       # print unsatisfied supply and demand
       nodes = sorted(self.node_commodity_data.index)
       for n in nodes:
          remaining_supply = self.primal.UnsatSupply[n].value
          if remaining_supply > 0:
             print('Remaining supply of %s on node %s: %.2f' %(str(n[1]), 
                str(n[0]), remaining_supply))
       for n in nodes:
          remaining_demand = self.primal.UnsatDemand[n].value
          if remaining_demand > 0:
             print('Remaining deman of %s on node %s: %.2f' %(str(n[1]), 
                str(n[0]), remaining_demand))

       # print final commodity flows
       for k in self.commodity_set:
          print('Flow Path for commodity: %s' %(str(k))
          for e0,e1 in self.arc_set:
             flow = self.primal.y[(e0, e1, k)].value
             if flow > 0:
             print('Flow on arc %s -> %s: %.2f %s'%(str(e0), 
                 str(e1), flow, str(k)))
        print()
        # top-level summary
        print('Number of attack(s) implemented: %d'%(self.total_attacks))
        print('Total cost with %d attack(s) = %.2f (primal) %.2f (dual)'%(self.attacks, self.primal.OBJ(), self.Idual.OBJ()))

########################
if __name__ == '__main__':
    m = MultiCommodityInterdiction('road_rail_network_nodes_data.csv',
                            'road_rail_network_nodes_commodity_data.csv',
                            'road_rail_network_arcs_data.csv',
                            'road_rail_network_arcs_commodity_data.csv')
    i = 0
    m.interdiction_scneario = 'Baseline'
    m.cost_scenario = 'Baseline'

    # keep going until we get no more benefit
    while i < float("inf"):
        m.attacks = i
        m.solve()
        m.printSolution()
       
        # stop if we don't use all attacks available
        if m.total_attacks < i: break 
        i = i+1

