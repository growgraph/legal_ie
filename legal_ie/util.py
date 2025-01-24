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


def visualize_nx(df, fname, scale=1, draw_edges=False):
    edges_labeled = [
        (*item[["subject", "object"]], {"relation": item["predicate"]})
        for _, item in df.iterrows()
    ]
    G = nx.DiGraph()
    G.add_edges_from(edges_labeled)

    _ = plt.figure(1, figsize=(scale * 30, scale * 15), dpi=300)

    pos = nx.nx_agraph.graphviz_layout(G, prog="neato", args="-Goverlap=false")

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

    if draw_edges:
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(fname + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fname + ".png", dpi=300, bbox_inches="tight")


def render_response(onto_str: str, text: str, llm):
    prompt = """You are an ontological triple extraction expert in criminal law.
        Please generate ontological triples in turtle (ttl) format from the text below.
        Be guided by the following extra ontology 

        ```ttl
        {ontology}
        ```

        Respect the following:

         - for entities that represent instances (specific to the document) use the namespace `@prefix cu: <https://growgraph.dev/current#> .`
         - make sure that resultant graph is connected with respect to `cu` namespace (all the entities specific to the document should have a path leading to the root, case id, for example)
         - while the input text is in French, make sure all generated literals are in English, except for the proper names
         - be as specific as possible, respect the granularity provided by the ontology <https://growgraph.dev/fcaont#>: 
            - map the age and social status of people involved
            - imperatively map offenses, punishments, appeal decisions to the ontology, if there are several, provide all of them            
         - represent durations as xsd:duration in ISO 8601 format. Double check ISO formatted entities.
         - feel free to propose to add new classes new and/or relations to <https://growgraph.dev/fcaont#>, if necessary for accuracy
         - pay attention to disambiguation: it is possible that the same entity is mentioned several times in the text, but only one instance should be added to semantic store 
         - provide all used namespaces in the header (all prefixes should be defined)
         - before submitting, check the validity of generated turtle snippet

        Please return two blocks:
        1. the extracted triples in a block marked as ```ttl ```
        2. A comments' block marked as ```remarks ```
        Remarks should be concise, itemized and ONLY concern additions and/or modifications to the provided ontology <https://growgraph.dev/fcaont#>.

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


def replace_namespaces(g, current_ns, new_ns):
    # SPARQL Update query to replace the namespace
    query = f"""
        DELETE {{
            ?s ?p ?o
        }}
        INSERT {{
            ?s_new ?p_new ?o_new
        }}
        WHERE {{
            ?s ?p ?o .
            BIND(IF(ISIRI(?s), IRI(REPLACE(STR(?s), "^{current_ns}", "{new_ns}")), ?s) AS ?s_new)
            BIND(IF(ISIRI(?p), IRI(REPLACE(STR(?p), "^{current_ns}", "{new_ns}")), ?p) AS ?p_new)
            BIND(IF(ISIRI(?o), IRI(REPLACE(STR(?o), "^{current_ns}", "{new_ns}")), ?o) AS ?o_new)
        }}
    """

    g.update(query)

    return g
