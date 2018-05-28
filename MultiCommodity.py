# example script to run the multi_commodity. Maybe transition this to a 
# command line script

if __name__ == '__main__':
    from multi_commodity import *

    node_data = 'sample_nodes_data.csv'
    node_commodity_data = 'sample_nodes_commodity_data.csv'
    arc_data = 'sample_arcs_data.csv'
    arc_commodity_data = 'sample_arcs_commodity_data.csv'
    arc_cost = 'ArcCost'
    interdiction_cost = 'InterdictionCost'
    outfile = '_outfile'

    m = MultiCommodityInterdiction(node_data,
                                   node_commodity_data,
                                   arc_data,
                                   arc_commodity_data,
                                   0,
                                   arc_cost,
                                   interdiction_cost,
                                   outfile)

    # add any desired heuristics here and run
    i = 0
    maxAttacks = 21
    delim = "_"

    # setup the output file writer
    csvfile = delim.join((outfile, arc_cost, interdiction_cost, "Totals.csv"))
    output = open(csvfile, "w+")
    writer = csv.writer(output, lineterminator="\n")

    # We're okay with re-writing headers, as someone may want to run
    # several scenarios using different files and output to a single file
    writer.writerows([["CostScenario", "InterdictionCase", "MaxAttacks", 
        "AttacksUsed", "ObjectiveValue"]])

    # substitute a heuristic here
    while i < maxAttacks:
        m.attacks = i
        m.solve()
        m.printSolution()
        writer.writerows([[m.arc_cost, m.interdiction_cost, m.attacks, 
            m.total_attacks, m.primal.OBJ()]])
        i = i + 1

    output.close()

