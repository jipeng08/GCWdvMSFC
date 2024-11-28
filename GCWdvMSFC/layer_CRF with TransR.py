# CRF with TransR layer knowledgement Extraction Model
import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pykeen
from pykeen.pipeline import pipeline
from pykeen.models import TransR
from pykeen.triples import TriplesFactory

# Step 1: Prepare the aluminum electrolyzer metrics data
data = {
    'average_voltage': [4.0, 4.1, 4.2, 4.0, 4.3],
    'electrolyte_level': [10.0, 9.8, 10.2, 9.9, 10.1],
    'fluoride_salt': [2.5, 2.6, 2.4, 2.5, 2.6],
    'anode_travel': [50.0, 55.0, 52.0, 53.0, 51.0],
    'aluminum_output': [100, 110, 105, 98, 103],
    'power_consumption': [500, 510, 520, 490, 515]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Step 2: Build the Knowledge Graph using RDFLib
g = Graph()
namespace = URIRef("http://example.org/")

for i, row in df.iterrows():
    # Create entities for each metric and value pair
    voltage = URIRef(f"{namespace}average_voltage_{i}")
    level = URIRef(f"{namespace}electrolyte_level_{i}")
    salt = URIRef(f"{namespace}fluoride_salt_{i}")
    anode = URIRef(f"{namespace}anode_travel_{i}")
    output = URIRef(f"{namespace}aluminum_output_{i}")
    power = URIRef(f"{namespace}power_consumption_{i}")

    # Add entities and their values to the graph
    g.add((voltage, RDF.type, RDFS.Class))
    g.add((voltage, URIRef(f"{namespace}has_value"), Literal(row['average_voltage'])))

    g.add((level, RDF.type, RDFS.Class))
    g.add((level, URIRef(f"{namespace}has_value"), Literal(row['electrolyte_level'])))

    g.add((salt, RDF.type, RDFS.Class))
    g.add((salt, URIRef(f"{namespace}has_value"), Literal(row['fluoride_salt'])))

    g.add((anode, RDF.type, RDFS.Class))
    g.add((anode, URIRef(f"{namespace}has_value"), Literal(row['anode_travel'])))

    g.add((output, RDF.type, RDFS.Class))
    g.add((output, URIRef(f"{namespace}has_value"), Literal(row['aluminum_output'])))

    g.add((power, RDF.type, RDFS.Class))
    g.add((power, URIRef(f"{namespace}has_value"), Literal(row['power_consumption'])))

# Print triples in the graph
print("Knowledge Graph Triples:")
for s, p, o in g:
    print(s, p, o)

# Step 3: Use CRF for Relationship Modeling
# Prepare training data for CRF
X_train = [[(key, row[key]) for key in row.index] for i, row in df.iterrows()]
y_train = [["has_value"] * len(row) for row in df.iterrows()]  # Simple label for relationships

# Define and train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, c2=0.1,  # Regularization parameters
    max_iterations=100
)
crf.fit(X_train, y_train)

# Predict the relationships for the training data
y_pred = crf.predict(X_train)

# Evaluate the CRF model
print("\nCRF Model Evaluation:")
print(metrics.flat_classification_report(y_train, y_pred))

# Step 4: Knowledge Graph Embedding using TransR (PyKEEN)
# Create triples from the knowledge graph for embedding
triples = [
    ('average_voltage_1', 'has_value', 4.0),
    ('electrolyte_level_1', 'has_value', 10.0),
    ('fluoride_salt_1', 'has_value', 2.5),
    ('anode_travel_1', 'has_value', 50.0),
    ('aluminum_output_1', 'has_value', 100),
    ('power_consumption_1', 'has_value', 500),
    # Add additional triples for other rows as needed
]

# Create a TriplesFactory object
triples_factory = TriplesFactory.from_labeled_triples(triples)

# Run the TransR model to learn embeddings
result = pipeline(
    model=TransR,
    dataset=triples_factory,
    training_loop='sls',  # Use Stochastic Local Search for optimization
    num_epochs=100,
)

# Get the embeddings for entities and relations
embeddings = result.model.entity_embeddings.weight.data
print("\nTransR Entity Embeddings:")
print(embeddings)

