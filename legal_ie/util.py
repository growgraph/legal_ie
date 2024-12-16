import io
import re

import pydotplus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# from IPython.display import display
from rdflib.tools.rdf2dot import rdf2dot

import networkx as nx
import matplotlib.pyplot as plt


def extract_struct(text, key):
    # Pattern to match text between ```key and ```
    pattern = rf"```{key}(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def visualize(g, path):
    stream = io.StringIO()
    # rdf2dot(g, stream, opts={display})
    rdf2dot(g, stream)
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    dg.write_png(path)


def visualize_nx(df, fname):
    edges_labeled = [
        (*item[["subject", "object"]], {"relation": item["predicate"]})
        for _, item in df.iterrows()
    ]
    G = nx.DiGraph()
    G.add_edges_from(edges_labeled)

    fig = plt.figure(1, figsize=(30, 15), dpi=300)

    pos = nx.nx_agraph.graphviz_layout(G)

    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # Draw nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=10,
        font_weight="bold",
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    fig.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")


def render_response(onto_str: str, text: str, llm):
    prompt = """You are an ontological triple extraction expert in criminal law.
        Please generate ontological triples in turtle (ttl) format from the text below.
        Use known namespaces such as schema.org.
        Be guided by the following extra ontology 

        ```ttl
        {ontology}
        ```

        For  and add them to <https://growgraph.dev/criminal#> (if not already present). 
        Be as specific as possible in parsing:

         - if there are many mentions, split them (e.g. several instances of Punishment)
         - represent durations as xsd:duration in ISO 8601 format
         - try to represent details and classify, where possible. E.g. prior convictions, types of offences etc
         - the input text is in French, but in response define types, classes and properties use English
         - do not forget to provide all used prefixes

        For previously unseen generic (specific) entities please use `@prefix cu: <https://growgraph.dev/current#> .`

        Input text
        ```
        {input_text}
        ```

        """

    parser = StrOutputParser()

    prompt = PromptTemplate(template=prompt, input_variables=["ontology", "input_text"])

    chain = prompt | llm | parser

    response = chain.invoke({"ontology": onto_str, "input_text": text})
    return response
