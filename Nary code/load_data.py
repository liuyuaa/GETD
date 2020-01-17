class Data:

    def __init__(self, data_dir="./data/WikiPeople-3/"):
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations if i not in self.train_relations] + [i for i in self.test_relations if i not in self.train_relations]
        self.relations = sorted(list(set(self.relations)))

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[0] for d in data])))
        return relations

    def get_entities(self, data):
        ent = []
        for k in range(1, len(data[0])):
            ent = ent + [d[k] for d in data]
        entities = sorted(list(set(ent)))
        return entities
