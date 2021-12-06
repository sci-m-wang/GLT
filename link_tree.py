from generate_tree import Tree

class LinkTree(object):
    '表示原型与当前样本的生成链接树，用于反事实的生成'
    def __init__(self, n_features, n_group_feature, prototype, instance):
        self.n_features = n_features
        self.n_group_feature = n_group_feature
        self.prototypes = prototype
        self.instance = instance
        self.root = Tree(None, self.n_group_feature)
        pass
    def gen_link(self):
        n_trees = self.n_features//self.n_group_feature-1
        trees_deepth = [0 for i in range(n_trees)]
        for i in range(len(trees_deepth)-1):
            trees_deepth[i] = self.n_group_feature
            pass
        trees_deepth[-1] = self.n_features % self.n_group_feature + self.n_group_feature
        # n_trees = self.n_features//self.n_group_feature
        # trees_deepth = [self.n_group_feature for i in range(n_trees)]
        to_tree = self.root
        while trees_deepth != []:
            deepth = trees_deepth.pop(0)
            protos = self.prototypes[:deepth]
            samples = self.instance[:deepth]
            self.prototypes = self.prototypes[deepth:]
            self.instance = self.instance[deepth:]
            opt, max_similarity = to_tree.gen_tree(protos, samples)
            next_tree = Tree((opt, max_similarity), deepth)
            to_tree.next = next_tree
            to_tree = to_tree.next
            pass
        pass
    def create_CF(self):
        counterfactual = []
        to_tree = self.root.next
        while to_tree.next != None:
            # print(to_tree.root.value)
            counterfactual += to_tree.root.value[0]
            to_tree = to_tree.next
            pass
        counterfactual += to_tree.root.value[0]
        opt, max_similarity = to_tree.gen_tree(self.prototypes,self.instance)
        counterfactual += opt
        return counterfactual



