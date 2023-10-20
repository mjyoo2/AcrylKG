import re
from pyvis.network import Network


def parse_parentheses(s):
    return re.findall(r'\((.*?)\)', s)

def graph_visualize(kg_output, output_path):
    kg_output = parse_parentheses(kg_output)
    triples = []
    for triple in kg_output:
        temp = triple.split(",", 2)
        print(temp)
        if len(temp) == 2:
            temp.append(" ")
            triples.append(temp)
        elif len(temp) == 3:
            triples.append(temp)

    # print(triples)
    # triples example
    # triples = [
    #     ('I', 'Am', 'Student'),
    #     ('I', 'Have', 'Phone'),
    #     ('Student', 'Use', 'Phone')
    # ]

    graph = Network()
    for s, p, o in triples:
        graph.add_node(s, label=s, size=20)
        graph.add_node(o, label=o, size=20)
        graph.add_edge(s, o, label=p, length=500)

    # save
    graph.save_graph(f"{output_path}/result.html")