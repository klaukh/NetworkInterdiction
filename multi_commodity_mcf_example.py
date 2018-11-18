
# example script to run the multi_commodity
import pandas as pd
import min_cost.flow as mcf

if __name__ == '__main__':

    # read in data and set parameters
    print('Reading in data...')
    node_data = pd.read_csv('sample_nodes_data.csv')
    node_commodity_data = pd.read_csv('sample_nodes_commodity_data.csv')
    arc_data = pd.read_csv('sample_arcs_data.csv')
    arc_commodity_data = pd.read_csv('sample_arcs_commodity_data.csv')

    # setup the object
    print('Creating LP...')
    m = mcf.MinCostFlow(nodes=node_data,
                       node_commodities=node_commodity_data,
                       arcs=arc_data,
                       arc_commodities=arc_commodity_data)

    # add any desired heuristics here and run
    print('Solving LP...')
    m.solve()
    m._primal.display()
    print('\n')
    print(m.objective_values)
    print(m.flows)
    print(m.arc_commodities)

