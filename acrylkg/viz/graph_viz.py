import re
from pyvis.network import Network


def parse_parentheses(s):
    return re.findall(r'\((.*?)\)', s)

def graph_visualize(kg_output, retrieval_KG):
    if type(kg_output) != list:
        kg_output = parse_parentheses(kg_output)
        triples = []
        for triple in kg_output:
            temp = triple.split(",", 2)
            print(temp)
            if len(temp) == 2:
                temp.insert(1, "is")
                triples.append(temp)
            elif len(temp) == 3:
                triples.append(temp)
    else:
        triples = kg_output

    if retrieval_KG is not None:
        retrieval_KG = parse_parentheses(", ".join(retrieval_KG))
        retrieval_triples = []
        for triple in retrieval_KG:
            temp = triple.split(",", 2)
            if len(temp) == 2:
                temp.insert(1, "is")
                retrieval_triples.append(temp)
            elif len(temp) == 3:
                retrieval_triples.append(temp)
    # print(triples)
    # triples example
    # triples = [
    #     ('I', 'Am', 'Student'),
    #     ('I', 'Have', 'Phone'),
    #     ('Student', 'Use', 'Phone')
    # ]

    graph = Network()
    for s, p, o in triples:
        color = "#D2E8FC"
        if retrieval_KG is not None:
            for kg in retrieval_triples:
                if s.strip() == kg[0].strip() and p.strip() == kg[1].strip() and o.strip() == kg[2].strip():
                    color = "#F25764"
        graph.add_node(s, label=s, size=20, color=color)
        graph.add_node(o, label=o, size=20, color=color)
        graph.add_edge(s, o, label=p, length=500, color=color)

    kg_html = graph.generate_html()
    kg_html = kg_html.replace("'", "\"")
    return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
         display-capture; encrypted-media;" sandbox="allow-modals allow-forms     allow-scripts allow-same-origin allow-popups
              allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""     allowpaymentrequest="" frameborder="0" srcdoc='{kg_html}'></iframe>""", triples