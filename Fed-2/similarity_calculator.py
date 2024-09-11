import numpy as np
from abc import ABCMeta, abstractmethod
from sentence_transformers import SentenceTransformer, util

def jaccard_similarity(list1, list2):
    if len(list1) == 0 or len(list2) == 0:
        return 0
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    if union != 0:
        result = round(intersection / union, 4)
    else:
        result = 0
    return result


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
def embed_similarity(rule_list1, rule_list2):
    embed_list1 =  model.encode(rule_list1)
    embed_list2 =  model.encode(rule_list2)
    if len(embed_list1) == 0 or len(embed_list2) == 0:
        return 0
    return round(util.cos_sim(embed_list1, embed_list2).mean().item(), 3)


class Similarity_Calculator(metaclass=ABCMeta):
    @abstractmethod
    def get_similarity(self):
        pass

    @abstractmethod
    def infer_similarity(self):
        pass


class Hashtags_Similarity_Calculator(Similarity_Calculator):
    # Get similarity between self and peers
    def get_similarity(self, self_data, data_pool):
        self_hashtags = getattr(self_data, 'hashtags')
        result_dict = {}
        for peer_ins in data_pool:
            if self_data is None or data_pool[peer_ins].hashtags == None:
                result_dict[peer_ins] = 0
            else:
                peer_hashtags = getattr(getattr(data_pool[peer_ins], 'hashtags'), 'data')
                similarity_result = jaccard_similarity(self_hashtags, peer_hashtags)
                result_dict[peer_ins] = similarity_result
        return result_dict

    # If do not have similarity, then infer similarity from other's similarity dict
    def infer_similarity(self, similarity_dict, data_pool):
        for ins in similarity_dict:
            if similarity_dict[ins] == 0:
                temp_result = []
                des_ins = ins

                for middle_ins in data_pool:
                    if middle_ins == ins:
                        continue
                    # If middle instance has simialrity data
                    if getattr(data_pool[middle_ins], 'similarity') is not None:
                        if (similarity_dict[middle_ins] != 0
                                and des_ins in getattr(data_pool[middle_ins], 'similarity').data.keys()
                                and getattr(data_pool[middle_ins], 'similarity').data[des_ins] != 0):
                            val_1 = similarity_dict[middle_ins]
                            val_2 = getattr(data_pool[middle_ins], 'similarity').data[des_ins]
                            temp_result.append(round((val_1 + val_2) / 2, 4))

                if len(temp_result) == 0:
                    continue
                else:
                    similarity_dict[ins] = round(np.mean(temp_result), 4)
        return similarity_dict


class Rules_Similarity_Calculator(Similarity_Calculator):
    # Get similarity between self and peers
    def get_similarity(self, self_data, data_pool):
        self_rules = getattr(self_data, 'rules')
        result_dict = {}
        for peer_ins in data_pool:
            if self_data is None or data_pool[peer_ins].rules == None:
                result_dict[peer_ins] = 0
            else:
                peer_rules = getattr(getattr(data_pool[peer_ins], 'rules'), 'data')
                similarity_result = embed_similarity(self_rules, peer_rules)
                result_dict[peer_ins] = similarity_result
        return result_dict

    # If do not have similarity, then infer similarity from other's similarity dict
    def infer_similarity(self, similarity_dict, data_pool):
        for ins in similarity_dict:
            if similarity_dict[ins] == 0:
                temp_result = []
                des_ins = ins

                for middle_ins in data_pool:
                    if middle_ins == ins:
                        continue
                    # If middle instance has simialrity data
                    if getattr(data_pool[middle_ins], 'similarity') is not None:
                        if (similarity_dict[middle_ins] != 0
                                and des_ins in getattr(data_pool[middle_ins], 'similarity').data.keys()
                                and getattr(data_pool[middle_ins], 'similarity').data[des_ins] != 0):
                            val_1 = similarity_dict[middle_ins]
                            val_2 = getattr(data_pool[middle_ins], 'similarity').data[des_ins]
                            temp_result.append(round((val_1 + val_2) / 2, 4))

                if len(temp_result) == 0:
                    continue
                else:
                    similarity_dict[ins] = round(np.mean(temp_result), 4)
        return similarity_dict


class Blocks_Similarity_Calculator(Similarity_Calculator):
    # Get similarity between self and peers
    def get_similarity(self, self_data, data_pool):
        self_blocks = getattr(self_data, 'blocks')
        result_dict = {}
        for peer_ins in data_pool:
            if self_data is None or data_pool[peer_ins].blocks == None:
                result_dict[peer_ins] = 0
            else:
                peer_blocks = getattr(getattr(data_pool[peer_ins], 'blocks'), 'data')
                similarity_result = jaccard_similarity(self_blocks, peer_blocks)
                result_dict[peer_ins] = similarity_result
        return result_dict

    # If do not have similarity, then infer similarity from other's similarity dict
    def infer_similarity(self, similarity_dict, data_pool):
        for ins in similarity_dict:
            if similarity_dict[ins] == 0:
                temp_result = []
                des_ins = ins

                for middle_ins in data_pool:
                    if middle_ins == ins:
                        continue
                    # If middle instance has simialrity data
                    if getattr(data_pool[middle_ins], 'similarity') is not None:
                        if (similarity_dict[middle_ins] != 0
                                and des_ins in getattr(data_pool[middle_ins], 'similarity').data.keys()
                                and getattr(data_pool[middle_ins], 'similarity').data[des_ins] != 0):
                            val_1 = similarity_dict[middle_ins]
                            val_2 = getattr(data_pool[middle_ins], 'similarity').data[des_ins]
                            temp_result.append(round((val_1 + val_2) / 2, 4))

                if len(temp_result) == 0:
                    continue
                else:
                    similarity_dict[ins] = round(np.mean(temp_result), 4)
        return similarity_dict


class Peers_Similarity_Calculator(Similarity_Calculator):
    # Get similarity between self and peers
    def get_similarity(self, self_data, data_pool):
        self_peers = getattr(self_data, 'peers')
        result_dict = {}
        for peer_ins in data_pool:
            if self_data is None or data_pool[peer_ins].peers == None:
                result_dict[peer_ins] = 0
            else:
                peer_peers = getattr(getattr(data_pool[peer_ins], 'peers'), 'data')
                similarity_result = jaccard_similarity(self_peers, peer_peers)
                result_dict[peer_ins] = similarity_result
        return result_dict

    # If do not have similarity, then infer similarity from other's similarity dict
    def infer_similarity(self, similarity_dict, data_pool):
        for ins in similarity_dict:
            if similarity_dict[ins] == 0:
                temp_result = []
                des_ins = ins

                for middle_ins in data_pool:
                    if middle_ins == ins:
                        continue
                    # If middle instance has simialrity data
                    if getattr(data_pool[middle_ins], 'similarity') is not None:
                        if (similarity_dict[middle_ins] != 0
                                and des_ins in getattr(data_pool[middle_ins], 'similarity').data.keys()
                                and getattr(data_pool[middle_ins], 'similarity').data[des_ins] != 0):
                            val_1 = similarity_dict[middle_ins]
                            val_2 = getattr(data_pool[middle_ins], 'similarity').data[des_ins]
                            temp_result.append(round((val_1 + val_2) / 2, 4))

                if len(temp_result) == 0:
                    continue
                else:
                    similarity_dict[ins] = round(np.mean(temp_result), 4)
        return similarity_dict