
# example script to run the multi_commodity
import pandas as pd
import min_cost.interdiction as mci

if __name__ == '__main__':

    # read in data and set parameters
    print('Reading in data...')
    node_data = pd.read_csv('sample_nodes_data.csv')
    node_commodity_data = pd.read_csv('sample_nodes_commodity_data.csv')
    arc_data = pd.read_csv('sample_arcs_data.csv')
    arc_commodity_data = pd.read_csv('sample_arcs_commodity_data.csv')
    arc_cost = 'ArcCost'
    interdiction_cost = 'InterdictionCost'

    # setup the object
    print('Creating LP...')
    m = mci.MinCostInterdiction(nodes=node_data,
                                   node_commodities=node_commodity_data,
                                   arcs=arc_data,
                                   arc_commodities=arc_commodity_data,
                                   attacks=0,
                                   arc_costs=arc_cost,
                                   interdiction_costs=interdiction_cost)

    # add any desired heuristics here and run
    print('Solving LP...')
    maxAttacks = 5
    for i in range(maxAttacks+1):
        m.attacks = i
        m.solve()
        print(m.flows)
        print(m.unsatisfied_commodities)

    # display the solver output for the last run
    m._primal.display()
    m._dual.display()
    print('\n')
    print(m.arc_commodities)
    print(m.objective_values)

