from flask import Flask, jsonify, request
import requests
import json
import numpy as np
from datetime import datetime
import pandas as pd
from collections import deque
from instance_data import get_peers, get_hashtags, get_blocks, get_rules, get_all
from packet_construction import get_packet
from send_receive_packet import send_packet, receive_packet, parse_received_packet
from FedApp_class import FedApp_Data, Peer_Selector, model_manager, NB, FedApp_Packet
from similarity_calculator import Hashtags_Similarity_Calculator
from peer_selector import select_peers
from gossip_controller import gossip_controller
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

###############################################
instance_name = 'mastodon.social'
instance_url = 'localhost:3000'
###############################################


received_buffer = deque()
data_pool = {}
stat = {'gossip_count': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
data = FedApp_Data()
selector = Peer_Selector()
calculator = Hashtags_Similarity_Calculator()
trainer = NB()

# Load training data and testing data
train_test_data = pd.read_feather('./train_test.feather')
train_test_data.rename(columns={'label_5': 'label'}, inplace=True)
train_test_data.drop(['label_8'], axis=1, inplace=True)

train_test_data = train_test_data.loc[train_test_data['instance'] == instance_name]
train_data = train_test_data.loc[(train_test_data['timestamp'] >= datetime.strptime('2023-03-03 00:00:00', '%Y-%m-%d %H:%M:%S')) & (train_test_data['timestamp'] < datetime.strptime('2023-05-01 00:00:00', '%Y-%m-%d %H:%M:%S'))]
train_data.reset_index(drop=True, inplace=True)

test_data = train_test_data.loc[(train_test_data['timestamp'] >= datetime.strptime('2023-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')) & (train_test_data['timestamp'] <= datetime.strptime('2023-06-20 23:59:59', '%Y-%m-%d %H:%M:%S'))]
test_data.reset_index(drop=True, inplace=True)

# @app.route('/get_data')
# @app.before_request
# @app.route('/')
# def get_instance_data():
#     with app.app_context():
#         global data
#         ins_data = get_all(instance_url)
#         # setattr(data, 'peers', ins_data['peers'])
#         setattr(data, 'peers', ['localhost:3001'])
#         setattr(data, 'hashtags', ins_data['hashtags'])
#         setattr(data, 'blocks', ins_data['block_list'])
#         setattr(data, 'rules', ins_data['rules'])
#         setattr(data, 'train_features', train_data)
#         setattr(data, 'test_features', test_data)
#         return jsonify({'message': 'Data updated successfully',
#                         'Peers length': len(getattr(data, 'peers')),
#                         'Hashtags length': len(getattr(data, 'hashtags')),
#                         'Rules length': len(getattr(data, 'rules'))})

@app.route('/')
def get_instance_data():
    with app.app_context():
        global data
        # ins_data = get_all(instance_name)
        setattr(data, 'peers', ['localhost:3001'])
        setattr(data, 'hashtags', ['abc', 'def', 'ghi'])
        setattr(data, 'train_features', train_data)
        setattr(data, 'test_features', test_data)
        return jsonify({'message': 'Data updated successfully',
                        'Peers length': len(getattr(data, 'peers')),
                        'Hashtags length': len(getattr(data, 'hashtags'))})


@app.route('/local_training')
def run_local_model():
    with app.app_context():
        global data
        global stat
        model, accuracy, precision, recall, f1 = trainer.train_model(getattr(data, 'train_features'))
        setattr(data, 'model', model)
        setattr(data, 'parameter', trainer.get_parameter(model))
        stat['accuracy'] = accuracy
        stat['precision'] = precision
        stat['recall'] = recall
        stat['f1'] = f1
        return jsonify({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})


# @app.route('/test_model')
# def run_test_model():
#     global data
#     pred, accuracy, precision, recall, f1 = trainer.evaluate_model(getattr(data, 'model'), getattr(data, 'test_features'))
#     temp_test_features = getattr(data, 'test_features')
#     temp_test_features['label'] = pred
#     warning_idx = temp_test_features.loc[temp_test_features['label'] == 1].index.tolist()
#     return jsonify({'warning_idx': warning_idx, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})



@app.route('/receive_packet', methods=['POST'])
def run_receive_packet():
    return receive_packet(received_buffer)


@app.route('/parse_received_packet')
def run_parse_received_packet():
    global received_buffer
    global data_pool
    received_buffer, data_pool = parse_received_packet(received_buffer, data_pool, instance_url)
    return str(data_pool)


@app.route('/gossip_hashtags')
def gossip_hashtags():
    with app.app_context():
        gossip_controller(instance_url, data, data_pool, selector, stat, 0, 1, 'hashtags')
        return jsonify({'state': 'Hashtags sent successfully'})


@app.route('/calculate_hashtags_similarity')
def calculate_hashtags_similarity():
    similarity_dict = calculator.get_similarity(data, data_pool)
    setattr(data, 'similarity', similarity_dict)

    infer_similarity_dict = calculator.infer_similarity(getattr(data, 'similarity'), data_pool)
    setattr(data, 'similarity', infer_similarity_dict)
    return jsonify({'state': 'Calculate similarity successfully'})


@app.route('/gossip_similarity')
def gossip_similarity():
    with app.app_context():
        gossip_controller(instance_url, data, data_pool, selector, stat, 0, 1, 'similarity')
        return jsonify({'state': 'Similarity sent successfully'})


@app.route('/gossip_parameter')
def gossip_parameter():
    with app.app_context():
        gossip_controller(instance_url, data, data_pool, selector, stat, 0, 1, 'parameter')
        return jsonify({'state': 'Parameter sent successfully'})


@app.route('/federal_learning')
def federal_learning():
    with app.app_context():
        new_parameter, weight_accuracy, weight_precision, \
        weight_recall, weight_f1, HCI = trainer.federal_learning(1, 0, True, data, data_pool, stat['f1'])
        # If success FL with peers and result better than current result
        if new_parameter is not None and weight_f1 > stat['f1']:
            setattr(data, 'parameter', new_parameter)
            stat['accuracy'] = weight_accuracy
            stat['precision'] = weight_precision
            stat['recall'] = weight_recall
            stat['f1'] = weight_f1
        setattr(data, 'HCI', HCI)

        return jsonify({'state': 'Federal learning successfully'})


@app.route('/show_result')
def show_result():
    # print('###############################################')
    # print('Data pool:', data_pool.keys())
    # print('Similarity:', getattr(data, 'similarity'))
    # print('###############################################')
    return jsonify({'Data pool': list(data_pool.keys()), 'Similarity': getattr(data, 'similarity'),
                    'Accuracy': stat['accuracy'], 'Precision': stat['precision'],
                    'Recall': stat['recall'], 'F1': stat['f1'], 'Gossip Count': stat['gossip_count']})


if __name__ == '__main__':
    # thread = Thread(target=run_parse_received_packet)
    # thread.start()

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_parse_received_packet, 'interval', seconds=1)
    # scheduler.add_job(gossip_hashtags, 'interval', seconds=5)
    # scheduler.add_job(calculate_hashtags_similarity, 'interval', seconds=5)
    # scheduler.add_job(show_result, 'interval', seconds=5)
    scheduler.start()

    get_instance_data()
    run_local_model()
    app.run(debug=True, port=3000)