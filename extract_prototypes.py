import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

class ProtoTypes():
    def __init__(self, datas, n_protos = 10):
        self.data = datas
        self.labels = None
        self.centers = None
        self.prototypes = []
        self.protos = None
        self.indices = []
        self.n_protos = n_protos
        pass
    def get_protos(self):
        kmeans = KMeans(init="k-means++", n_clusters=self.n_protos, n_init=4, random_state=0)
        kmeans.fit(self.data)
        self.centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        for each in self.centers:
            min_dist = np.inf
            center = each
            i = 0
            for item in self.data:
                dist = np.sum(1-(each == item))
                if dist < min_dist:
                    min_dist = dist
                    center = item
                    index = i
                    # self.indices.append(i)
                    pass
                # self.indices.append(i)
                i += 1
                pass
            self.indices.append(index)
            self.prototypes.append(center)
            pass
        proto_types = np.array(self.prototypes)
        self.protos = pd.DataFrame(proto_types)
        return self.protos, self.indices
    def get_prototypes(self):
        self.get_protos()
        # if self.protos == None:
        #     raise NotImplementedError("must run function \"get_protos\" first.")
        # else:
        #     return self.prototypes
        return self.prototypes, self.indices
    pass

