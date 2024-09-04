import csv


class GENAKnowledgeGraph:
    def __init__(self, csv_file):
        self.graph = {}
        self.load_graph(csv_file)

    def load_graph(self, csv_file):
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                subject, relation, obj = row
                if subject not in self.graph:
                    self.graph[subject] = {}
                if relation not in self.graph[subject]:
                    self.graph[subject][relation] = []
                self.graph[subject][relation].append(obj)

    def retrieve_related_entities(self, entity):
        if entity in self.graph:
            return self.graph[entity]
        else:
            return None

    def expand_query(self, query):
        expanded_query = set()
        if query in self.graph:
            # Consider direct relations
            for relation, objects in self.graph[query].items():
                expanded_query.update(objects)

                # Consider indirect relations through related entities
                for related_entity in objects:
                    if related_entity in self.graph:
                        for related_relation, related_objects in self.graph[
                            related_entity
                        ].items():
                            expanded_query.update(related_objects)
        return expanded_query


gena_kg = GENAKnowledgeGraph("gena_data_final_triples.csv")

entity = "vitamin K"  # will work as Vitamin K is present
# entity = "vitamin X" # wont work as Vitamin X is not present

related_entities = gena_kg.retrieve_related_entities(entity)
if related_entities:
    print(f"Related entities for '{entity}':")
    for relation, objects in related_entities.items():
        print(f"{relation}: {', '.join(objects)}")
else:
    print(f"No information found for '{entity}' in the knowledge graph.")

expanded_entities = gena_kg.expand_query(entity)

print("Original Query:", entity)
print("Expanded Query:")
for entity in expanded_entities:
    print(entity)
