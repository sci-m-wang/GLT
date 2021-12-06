import numpy as np
import Levenshtein

class Node(object):
    def __init__(self, value, type):
        self.value = value
        self.left = None
        self.right = None
        self.type = type
        # self.childs = []
        self.path = []
        self.parent = None
        pass
    def copy(self):
        new_node = Node(self.value,self.type)
        new_node.left = self.left
        new_node.right = self.right
        return new_node
    def get_path(self):
        to_node = self
        while to_node.parent:
            self.path += [to_node]
            to_node = to_node.parent
            pass
        pass

    pass


class Tree(object):
    '基本树结构'
    def __init__(self, root, deepth):
        self.root = Node(value=root, type='root')
        self.max_deepth = deepth
        self.deepth = 0
        self.paths = []
        self.next = None
        pass
    def add_nodes(self, protos = [Node(0,'proto')], samples = [Node(1,'sample')]):                   # 需要按照原型与样本对应的顺序给定节点列表
        q = [[self.root]]
        count = 0
        # print(protos)
        while self.deepth < self.max_deepth:
            pop_node = q[self.deepth][count]
            if pop_node.left != None and pop_node.right != None:
                q.append([])
                q[self.deepth+1].append(pop_node.left)
                q[self.deepth+1].append(pop_node.right)
                count += 1
                if count >= len(q[self.deepth]):                    # 左右子节点都非空且count等于该层节点数时，说明该层已经填满
                    self.deepth += 1
                    count = 0
                continue
            elif pop_node.left == None:
                # print(protos,len(protos))
                new_node = protos[self.deepth].copy()
                pop_node.left = new_node
                new_node.parent = pop_node
                # pop_node.childs.append(new_node)
                new_node.get_path()
                self.paths.append(new_node.path)
                # print("Added proto node sucessfully.")
            elif pop_node.right == None:
                new_node = samples[self.deepth].copy()
                pop_node.right = new_node
                new_node.parent = pop_node
                # pop_node.childs.append(new_node)
                new_node.get_path()
                self.paths.append(new_node.path)
                # print("Added sample node sucessfully.")
            else:
                raise NotImplementedError("INSRT ERROR. There has been a node at the location will be inserted.")
            pass
        # print(self.leaves)
        pass
    def gen_leaves(self):
        paths = []
        for each in self.paths:
            if len(each) == self.max_deepth:
                paths.append(each)
                pass
            pass
        self.paths = paths
        # print(len(self.paths))
        pass
    def get_leaves(self):
        leaves = []
        i = 0
        for each in self.paths:
            leaves.append([])
            for item in each:
                leaves[i].append(item.value)
                pass
            i += 1
            pass
        return leaves
    def gen_opt(self):
        protos = []
        sample = []
        leaves = self.get_leaves()
        opt = None  # 记录局部最优结果
        to_node = self.root
        while to_node.left != None:  # 获取最左侧为原型
            protos.append(to_node.left.value)
            to_node = to_node.left
            pass
        to_node = self.root
        while to_node.right != None:
            sample.append(to_node.right.value)
            to_node = to_node.right
            pass
        max_similarity = 0
        for each in leaves:
            each = np.array(each)
            protos = np.array(protos)
            sample = np.array(sample)
            cost = 1/(1 + np.exp(np.sqrt(np.sum((each - sample) ** 2))/np.sqrt(np.sqrt(np.sum(each ** 2))*np.sqrt(np.sum(sample ** 2)))))  # 变为当前结果的代价
            simi = np.exp(-np.sqrt(np.sum((each-protos) ** 2))/np.sqrt(np.sqrt(np.sum(each ** 2))*np.sqrt(np.sum(protos ** 2))))  # 当前结果与原型的相似度
            relative_similarity = simi / cost
            # print(sample,protos)
            # print(cost,simi,relative_similarity)
            if relative_similarity >= max_similarity:
                max_similarity = relative_similarity
                opt = each
                pass
            pass
        return opt, max_similarity
    def gen_tree(self, protos = [Node(0,'proto')], samples = [Node(1,'sample')]):
        self.add_nodes(protos,samples)
        self.gen_leaves()
        self.get_leaves()
        opt, max_similarity = self.gen_opt()
        return opt.tolist(), max_similarity


