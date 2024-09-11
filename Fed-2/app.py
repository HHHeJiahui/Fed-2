from flask import Flask, jsonify, request
import requests
import json
import swifter
import sys
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
from collections import deque
from instance_data import get_peers, get_hashtags, get_blocks, get_rules, get_all
from packet_construction import get_packet
from send_receive_packet import send_packet, receive_packet, parse_received_packet
from FedApp_class import FedApp_Data, Peer_Selector, model_manager, NB, LR, FedApp_Packet
from similarity_calculator import Hashtags_Similarity_Calculator, Blocks_Similarity_Calculator, Rules_Similarity_Calculator, Peers_Similarity_Calculator
from peer_selector import select_peers
from gossip_controller import gossip_controller
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)



##############################################################
Mastodon_File_Path = '/home/hejiahui/mastodon/'
instance_url = 'mastodon.social'
instance_name = 'localhost:5000'
prelabeled_data = pd.read_feather(f'{Mastodon_File_Path}/FedApp/mastodon_social.feather')
prelabeled_data = prelabeled_data.iloc[:100]
local_model_path = ''
calculator = Hashtags_Similarity_Calculator()
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
##############################################################




####### Setup parameters from .rake command #######
args = sys.argv[1:]

if args[0] == 'NB':
    trainer = NB()
elif args[0] == 'LR':
    trainer = LR()
else:
    trainer = NB()

if args[1] == 'Hashtags':
    calculator = Hashtags_Similarity_Calculator()
elif args[1] == 'Peers':
    calculator = Peers_Similarity_Calculator()
elif args[1] == 'Blocks':
    calculator = Blocks_Similarity_Calculator()
elif args[1] == 'Rules':
    calculator = Rules_Similarity_Calculator()
else:
    calculator = Hashtags_Similarity_Calculator()

if float(args[2]) > 1:
    gossip_peer_num = int(args[2])
    gossip_peer_frac = 0
else:
    gossip_peer_num = 0
    gossip_peer_frac = float(args[2])

gossip_data_time = int(args[3])
calculate_similarity_time = int(args[4])
federated_partner_num = int(args[5])
federated_learning_time = int(args[6])


####### Define some local data #######
data = FedApp_Data()
data_pool = {}
selector = Peer_Selector()
received_buffer = deque()
stat = {'gossip_count': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

####### Load training data and testing data #######
def get_embed(row):
    return embed_model.encode(row['feature'])
prelabeled_data['feature'] = prelabeled_data.swifter.apply(get_embed, axis=1)

X_train, X_test, y_train, y_test = train_test_split(prelabeled_data['feature'], prelabeled_data['label'], test_size=0.2, random_state=42)
train_data = pd.DataFrame({'feature': X_train, 'label': y_train})
test_data = pd.DataFrame({'feature': X_test, 'label': y_test})


@app.route('/get_data')
@app.before_request
def get_instance_data():
    with app.app_context():
        global data
        ins_data = get_all(instance_url)
        setattr(data, 'peers', ins_data['peers'])
        # setattr(data, 'peers', ['localhost:3001'])
        setattr(data, 'hashtags', ins_data['hashtags'])
        setattr(data, 'blocks', ins_data['block_list'])
        setattr(data, 'rules', ins_data['rules'])
        setattr(data, 'train_features', train_data)
        setattr(data, 'test_features', test_data)
        return jsonify({'message': 'Data load successfully',
                        'Peers length': len(getattr(data, 'peers')),
                        'Hashtags length': len(getattr(data, 'hashtags')),
                        'Rules length': len(getattr(data, 'rules'))})


# @app.route('/get_data')
# def get_instance_data():
#     with app.app_context():
#         global data
#         ins_data = get_all(instance_url)
#         setattr(data, 'peers', ['localhost:3001'])
#         setattr(data, 'hashtags', ins_data['hashtags'])
#         return jsonify({'message': 'Data updated successfully',
#                         'Peers length': len(getattr(data, 'peers')),
#                         'Hashtags length': len(getattr(data, 'hashtags'))})


@app.route('/train_local_model', methods=['GET'])
def train_local_model():
    with app.app_context():
        global data
        global stat

        # If there is no exist local model, then train a new one
        if local_model_path == '':
            model, accuracy, precision, recall, f1 = trainer.train_model(getattr(data, 'train_features'))
            setattr(data, 'model', model)
            setattr(data, 'parameter', trainer.get_parameter(model))
            stat['accuracy'] = accuracy
            stat['precision'] = precision
            stat['recall'] = recall
            stat['f1'] = f1
            return jsonify({'state': 'Train a new local model', 'accuracy': accuracy,
                            'precision': precision, 'recall': recall, 'f1': f1})

        # If there is a local model, then import that model into FedApp
        else:
            model = joblib.load(local_model_path)
            setattr(data, 'model', model)
            setattr(data, 'parameter', trainer.get_parameter(model))
            pred, accuracy, precision, recall, f1 = trainer.evaluate_model(getattr(data, 'model'),
                                                                           getattr(data, 'test_features'))
            stat['accuracy'] = accuracy
            stat['precision'] = precision
            stat['recall'] = recall
            stat['f1'] = f1
            return jsonify({'state': 'Import local model', 'accuracy': accuracy,
                            'precision': precision, 'recall': recall, 'f1': f1})


@app.route('/receive_packet', methods=['POST'])
def run_receive_packet():
    return receive_packet(received_buffer)


@app.route('/parse_received_packet')
def run_parse_received_packet():
    global received_buffer
    global data_pool
    received_buffer, data_pool = parse_received_packet(received_buffer, data_pool, instance_name)
    return str(data_pool)


@app.route('/gossip_local_data')
def gossip_local_data():
    with app.app_context():
        gossip_controller(instance_name, data, data_pool, selector, stat, gossip_peer_num, gossip_peer_frac, args[1])
        return jsonify({'state': 'Hashtags sent successfully'})

@app.route('/gossip_similarity')
def gossip_similarity():
    with app.app_context():
        gossip_controller(instance_name, data, data_pool, selector, stat, gossip_peer_num, gossip_peer_frac, 'similarity')
        return jsonify({'state': 'Similarity sent successfully'})

@app.route('/gossip_parameter')
def gossip_parameter():
    gossip_controller(instance_name, data, data_pool, selector, stat, gossip_peer_num, gossip_peer_frac, 'parameter')
    return "Parameter sent successfully"

@app.route('/calculate_similarity')
def calculate_similarity():
    similarity_dict = calculator.get_similarity(data, data_pool)
    setattr(data, 'similarity', similarity_dict)

    infer_similarity_dict = calculator.infer_similarity(getattr(data, 'similarity'), data_pool)
    setattr(data, 'similarity', infer_similarity_dict)
    return jsonify({'state': 'Calculate similarity successfully'})

@app.route('/federal_learning')
def federal_learning():
    global data
    new_parameter, new_accuracy, new_precision, \
    new_recall, new_f1, HCI = trainer.federal_learning(federated_partner_num, False,
                                                       getattr(data, 'parameter'), stat['f1'],
                                                       getattr(data, 'similarity'), getattr(data, 'HCI'),
                                                       getattr(data, 'test_features'), data_pool)

    new_model = trainer.set_patameter(new_parameter)
    setattr(data, 'model', new_model)
    setattr(data, 'parameter', new_parameter)
    setattr(data, 'HCI', HCI)
    stat['accuracy'] = new_accuracy
    stat['precision'] = new_precision
    stat['recall'] = new_recall
    stat['f1'] = new_f1
    return "FL successfully"

@app.route('/test_model')
def run_test_model():
    global data
    pred, accuracy, precision, recall, f1 = trainer.evaluate_model(getattr(data, 'model'), getattr(data, 'test_features'))
    temp_test_features = getattr(data, 'test_features')
    temp_test_features['label'] = pred
    warning_idx = temp_test_features.loc[temp_test_features['label'] == 1].index.tolist()
    return jsonify({'warning_idx': warning_idx, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})


# @app.route('/show_result')
# def show_result():
#     # print('###############################################')
#     # print('Data pool:', data_pool.keys())
#     # print('Similarity:', getattr(data, 'similarity'))
#     # print('###############################################')
#     return jsonify({'Data pool': list(data_pool.keys()), 'Similarity': getattr(data, 'similarity')})


if __name__ == '__main__':
    # thread = Thread(target=run_parse_received_packet)
    # thread.start()

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_parse_received_packet, 'interval', seconds=5)
    scheduler.add_job(gossip_local_data, 'interval', seconds=gossip_data_time)
    scheduler.add_job(gossip_similarity, 'interval', seconds=gossip_data_time)
    scheduler.add_job(gossip_parameter, 'interval', seconds=gossip_data_time)
    scheduler.add_job(calculate_similarity, 'interval', seconds=calculate_similarity_time)
    scheduler.add_job(federal_learning, 'interval', seconds=federated_learning_time)
    scheduler.start()

    get_instance_data()
    train_local_model()
    print(stat)
    app.run(debug=True, port=5000)