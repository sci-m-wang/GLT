import numpy as np
from link_tree import LinkTree
from generate_tree import Node
from extract_prototypes import ProtoTypes
from tqdm import tqdm

class Creator(object):
    def __init__(self,model,goals,samples,n_features,deepth,target_label,n_CFs):
        self.goals = goals
        self.n_CFs = n_CFs
        Proto = ProtoTypes(self.goals,self.n_CFs)
        self.protos = Proto.get_prototypes()
        self.samples = samples
        self.model = model
        self.n_features = n_features
        self.deepth = deepth
        self.target_label = target_label

        # 初始化节点类型
        self.SampleNodes = []
        self.ProtoNodes = []
        for each in self.samples:
            sample_node = []
            for value in each:
                sample_node.append(Node(value,"sample"))
                pass
            self.SampleNodes.append(sample_node)
            pass
        for item in self.samples:
            proto_node = []
            for e in item:
                proto_node.append(Node(e,"proto"))
                pass
            self.ProtoNodes.append(proto_node)
            pass
        self.Counterfactuals = []
        pass
    def createCFs(self):
        for each in tqdm(self.SampleNodes):
            for proto in self.ProtoNodes:
                tree = LinkTree(self.n_features,self.deepth,proto,each)
                tree.gen_link()
                counterfactual = tree.create_CF()
                if self.model.predict(np.array(counterfactual).reshape(1,-1)) == self.target_label:
                    self.Counterfactuals.append(tuple(counterfactual+[self.target_label]))
                    pass
                pass
            pass
        return self.Counterfactuals
    pass