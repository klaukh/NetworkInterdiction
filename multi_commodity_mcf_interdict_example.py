
# example script to run the multi_commodity
import pandas as pd
import min_cost.interdiction as mc

if __name__ == '__main__':

    # read in data and set parameters
    node_data = pd.read_csv('sample_nodes_data.csv')
    node_commodity_data = pd.read_csv('sample_nodes_commodity_data.csv')
    arc_data = pd.read_csv('sample_arcs_data.csv')
    arc_commodity_data = pd.read_csv('sample_arcs_commodity_data.csv')
    arc_cost = 'ArcCost'
    interdiction_cost = 'InterdictionCost'

    # setup the object
    m = mc.MinCostInterdiction(nodes=node_data,
                                   node_commodities=node_commodity_data,
                                   arcs=arc_data,
                                   arc_commodities=arc_commodity_data,
                                   attacks=0,
                                   arc_costs=arc_cost,
                                   interdiction_costs=interdiction_cost)


    # add any desired heuristics here and run
    maxAttacks = 5
    for i in range(maxAttacks+1):
        m.attacks = i
        m.solve()

    print(m.objective_values)
    print(m.flows)
    print(m.arc_commodities)

