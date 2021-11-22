import copy
from copy import deepcopy
from collections import defaultdict
from dataset import *
import numpy as np
from torch.utils.data.sampler import Sampler


class CustomSampler(Sampler):
    def __init__(self, data, num_classes, num_instances):
        self.data_source = data
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.pid_to_index_dict = defaultdict(list)
        self.index_to_pid_dict = defaultdict(list)  # usato per i test
        self.length = 0
        for index, (_, pid, _) in enumerate(self.data_source):
            self.pid_to_index_dict[pid].append(index)
            self.index_to_pid_dict[index].append(pid)  # usato per i test
        self.pids = list(self.pid_to_index_dict.keys())

    def __iter__(self):
        pid_to_batch = defaultdict(list)

        for pid in self.pids:
            # prendo gli indici per ogni identità
            indices = copy.deepcopy(self.pid_to_index_dict[pid])

            # se una classe non ha abbastanza esempi
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True)

            # shuffle così gli stessi indici non finiscono sempre nello stesso batch
            random.shuffle(indices)
            single_id_batch = []
            for index in indices:
                single_id_batch.append(index)
                if len(single_id_batch) == self.num_instances:
                    pid_to_batch[pid].append(single_id_batch)
                    single_id_batch = []

        available_ids = deepcopy(self.pids)

        batches = []
        while len(available_ids) >= self.num_classes:
            # prendo num_classes pids per random per batch
            batch_pids = random.sample(available_ids, k=self.num_classes)
            batch = []
            for pid in batch_pids:
                if len(pid_to_batch[pid]) == 0:
                    available_ids.remove(pid)
                    continue
                batch.extend(pid_to_batch[pid][0])
                pid_to_batch[pid].pop(0)
            batches.append(batch)
            # print(batches)
            # for el in batches:
            #    print(self.index_to_pid_dict[el])
            # exit()
        # print("batches", batches)
        self.length = len(batches)
        return iter(batches)

    def __len__(self):
        return self.length
