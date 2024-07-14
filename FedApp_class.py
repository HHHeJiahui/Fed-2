import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB


# Data_pool:
#     --- instance_name
#         --- FedApp_Data
#             --- FedApp_Packet


class FedApp_Data():
    def __init__(self):
        self.peers = None
        self.hashtags = None
        self.blocks = None
        self.rules = None
        self.toots = None
        self.train_features = None
        self.test_features = None
        self.model = None
        self.parameter = None
        self.similarity = {}
        self.HCI = {}


class FedApp_Packet():
    def __init__(self, src_ins, dest_ins, data, data_type):
        self.src_ins = src_ins
        self.dest_ins = dest_ins
        self.time = time.time()
        self.TTL = 3
        self.data = data
        self.data_type = data_type
        self.gossip_path = [src_ins]


class Peer_Selector():
    def select_n_random_peers(self, peer_list, n=0):
        if peer_list is None or len(peer_list) == 0:
            print('No peer list!')
        if n >= 0:
            if n <= len(peer_list):
                return random.sample(peer_list, n)
            else:
                return peer_list
        else:
            print('Error: n must larger or equal than 0')

    def select_frac_random_peers(self, peer_list, frac=0):
        if peer_list is None or len(peer_list) == 0:
            print('No peer list!')
        if frac >= 0 and frac <= 1:
            return random.sample(peer_list, round(len(peer_list) * frac))
        else:
            print('Error: frac must follow 0 <= frac <= 1')


class model_manager(metaclass=ABCMeta):
    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def get_parameter(self):
        pass

    @abstractmethod
    def merge_parameter(self):
        pass

    @abstractmethod
    def federal_learning(self):
        pass


class NB(model_manager):

    def train_model(self, train_features):
        # If current training data only have one label
        try:
            if train_features is None:
                print('No training features')
                return None, -1, -1, -1, -1

            if len(train_features['label'].value_counts()) == 1:
                print('Only have one label')
                return None, -1, -1, -1, -1

            if min(train_features['label'].value_counts()) == 1:
                print('Minor label less than 2')
                return None, -1, -1, -1, -1
        except:
            return None, -1, -1, -1, -1

        #         print(len(train_features))

        curr_features = np.vstack(train_features['embed'])
        curr_labels = list(train_features['label'])

        train_feature, test_features, train_label, test_labels = train_test_split(curr_features, curr_labels,
                                                                                  test_size=0.2, random_state=42,
                                                                                  stratify=curr_labels)

        nb = GaussianNB()
        parameters = {
            'var_smoothing': np.logspace(0, -9, num=40)
        }
        cv = GridSearchCV(nb, parameters, cv=5)
        cv.fit(train_feature, train_label)

        pred = cv.best_estimator_.predict(test_features)
        accuracy = round(accuracy_score(test_labels, pred), 3)
        precision = round(precision_score(test_labels, pred, average="macro"), 3)
        recall = round(recall_score(test_labels, pred, average="macro"), 3)
        f1 = round(f1_score(test_labels, pred, average="macro"), 3)

        return cv.best_estimator_, accuracy, precision, recall, f1

    def evaluate_model(self, curr_model, test_features):
        if curr_model is None or test_features is None:
            print('No local model or no test features')
            return -1, -1, -1, -1
        if len(test_features) == 0:
            print('No testing data')
            return -1, -1, -1, -1

        curr_features = np.vstack(test_features['embed'])
        curr_labels = list(test_features['label'])

        pred = curr_model.predict(curr_features)
        accuracy = round(accuracy_score(curr_labels, pred), 3)
        precision = round(precision_score(curr_labels, pred, average="macro"), 3)
        recall = round(recall_score(curr_labels, pred, average="macro"), 3)
        f1 = round(f1_score(curr_labels, pred, average="macro"), 3)

        return accuracy, precision, recall, f1

    def get_parameter(self, local_model):
        if local_model is None:
            print('Local model is None!')
            return None

        temp_prior = local_model.class_prior_
        temp_theta = local_model.theta_
        temp_var = local_model.var_
        temp_epsilon = local_model.epsilon_

        return [temp_prior, temp_theta, temp_var, temp_epsilon]

    def merge_parameter(self, local_parameter, target_parameter):
        if local_parameter is None and target_parameter is None:
            print('Both model is None!')
            return None
        if local_parameter is None and target_parameter is not None:
            return target_parameter
        if local_parameter is not None and target_parameter is None:
            return local_parameter

        avg_coef = np.mean([local_parameter[0], target_parameter[0]], axis=0)
        avg_intercept = np.mean([local_parameter[1], target_parameter[1]], axis=0)

        #         lr_avg = LogisticRegression()
        #         lr_avg.coef_ = avg_coef
        #         lr_avg.intercept_ = avg_intercept
        #         lr_avg.classes_ = np.array([0., 1.])

        return [avg_coef, avg_intercept]

    def federal_learning(self, n, thres, use_random, local_data, data_pool, local_f1):

        local_parameter = getattr(local_data, 'parameter')
        similarity_dict = getattr(local_data, 'similarity')
        test_features = getattr(local_data, 'test_features')
        HCI = getattr(local_data, 'HCI')

        # # Some error cases
        # if n < 0 or thres < 0 or thres > 1:
        #     print('Number incorrect! Follow rule: either n > 0 or 0 < thres < 1.')
        #     return None, None, None, None, None, None
        # if random == True and thres > 0:
        #     print('Cannot set threshold for random strategy.')
        #     return None, None, None, None, None, None
        # if local_parameter is None or similarity_dict is None or data_pool is None or test_features is None:
        #     print('Either local parameter simialrity dict or data pool or test features is None!')
        #     return None, None, None, None, None, None

        # Select federal peers
        federal_ins = []
        if n > 0 and thres == 0:
            if use_random == True:
                data_pool_ins = []
                for item in data_pool:
                    if getattr(data_pool[item], 'parameter') is not None:
                        data_pool_ins.append(item)
                if n <= len(data_pool_ins):
                    federal_ins = random.sample(data_pool_ins, n)
                else:
                    federal_ins = data_pool_ins
            else:
                sort_dict = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
                sort_ins_list = list(dict(sort_dict).keys())
                federal_ins = sort_ins_list[:n]
        elif n == 0 and thres > 0:
            federal_ins = list({k: v for k, v in similarity_dict.items() if v >= thres}.keys())
        else:
            print('Number incorrect! Follow rule: either n > 0 or 0 < thres < 1.')
            return None, None, None, None, None, None

        # Error case: if cannot select peers
        if len(federal_ins) == 0:
            print('Cannot select eligible peers!')
            return None, None, None, None, None, None

        # Step 1: Pairwise federated evaluation, to find which are useful, which are useless
        # Step 2: Weighted federated learning, only federal with useful peers
        # Step 3: Return new parameter and new performance, as well as result of each peers

        # Step 1: Pairwise federated evaluation
        pairs_fl_result = {}
        for ins in federal_ins:
            if getattr(data_pool[ins], 'parameter') is not None:
                prior_list = [local_parameter[0], getattr(getattr(data_pool[ins], 'parameter'), 'data')[0]]
                theta_list = [local_parameter[1], getattr(getattr(data_pool[ins], 'parameter'), 'data')[1]]
                var_list = [local_parameter[2], getattr(getattr(data_pool[ins], 'parameter'), 'data')[2]]
                epsilon_list = [local_parameter[3], getattr(getattr(data_pool[ins], 'parameter'), 'data')[3]]

                prior_avg = np.mean(np.array(prior_list), axis=0)
                theta_avg = np.mean(np.array(theta_list), axis=0)
                var_avg = np.mean(np.array(var_list), axis=0)
                new_epsilon = max(epsilon_list)

                nb_avg = GaussianNB()
                nb_avg.class_prior_ = prior_avg
                nb_avg.theta_ = theta_avg
                nb_avg.var_ = var_avg
                nb_avg.epsilon_ = new_epsilon
                nb_avg.classes_ = np.array([0., 1.])

                temp_accuracy, temp_precision, temp_recall, temp_f1 = self.evaluate_model(nb_avg, test_features)

                pairs_fl_result[ins] = round(temp_f1 - local_f1, 3)

        # Step 2: Weighted federated learning
        weighted_prior_list = [local_parameter[0]]
        weighted_theta_list = [local_parameter[1]]
        weighted_var_list = [local_parameter[2]]
        weighted_epsilon_list = [local_parameter[3]]
        weight_list = [1]

        for ins in federal_ins:
            if getattr(data_pool[ins], 'parameter') is not None:
                temp_prior = getattr(getattr(data_pool[ins], 'parameter'), 'data')[0]
                temp_theta = getattr(getattr(data_pool[ins], 'parameter'), 'data')[1]
                temp_var = getattr(getattr(data_pool[ins], 'parameter'), 'data')[2]
                temp_epsilon = getattr(getattr(data_pool[ins], 'parameter'), 'data')[3]

                if len(HCI) != 0 and ins in HCI.keys():
                    temp_weight = HCI[ins]
                else:
                    temp_weight = 0.5

                weighted_prior_list.append(temp_prior * temp_weight)
                weighted_theta_list.append(temp_theta * temp_weight)
                weighted_var_list.append(temp_var * temp_weight)
                weighted_epsilon_list.append(temp_epsilon)
                weight_list.append(temp_weight)

        weighted_prior = sum(weighted_prior_list) / sum(weight_list)
        weighted_theta = sum(weighted_theta_list) / sum(weight_list)
        weighted_var = sum(weighted_var_list) / sum(weight_list)
        weighted_epsilon = max(weighted_epsilon_list)

        nb_weight = GaussianNB()
        nb_weight.class_prior_ = weighted_prior
        nb_weight.theta_ = weighted_theta
        nb_weight.var_ = weighted_var
        nb_weight.epsilon_ = weighted_epsilon
        nb_weight.classes_ = np.array([0., 1.])

        weight_accuracy, weight_precision, weight_recall, weight_f1 = self.evaluate_model(nb_weight, test_features)
        new_parameter = [weighted_prior, weighted_theta, weighted_var, weighted_epsilon]

        # Update HCI
        for ins in pairs_fl_result:
            if len(HCI) != 0 and ins in HCI.keys():
                temp_result = HCI[ins] + (pairs_fl_result[ins] * 0.5)
                if temp_result > 1:
                    temp_result = 1
                elif temp_result < 0:
                    temp_result = 0
                HCI[ins] = temp_result
            else:
                HCI[ins] = 0.5 + (pairs_fl_result[ins] * 0.5)

        print(weight_f1)
        # Step 3
        return new_parameter, weight_accuracy, weight_precision, weight_recall, weight_f1, HCI