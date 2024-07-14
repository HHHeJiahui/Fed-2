from flask import jsonify, request
import requests
import json
from FedApp_class import FedApp_Data, FedApp_Packet
import pickle


# Send packet to peer
def send_packet(peer, packet):
    headers = {'Content-Type': 'application/json'}
    target_ins_url = f'http://{peer}/receive_packet'
    try:
        response = requests.post(target_ins_url, data=pickle.dumps(packet), headers=headers)
        response.raise_for_status()
        return jsonify({'status': 'Data sent to peer', 'response': response})
    except requests.exceptions.HTTPError as err:
        return jsonify({'status': 'HTTP error occurred', 'error': str(err)}), 400
    except Exception as err:
        return jsonify({'status': 'An error occurred', 'error': str(err)}), 500


# Receive packet from peer, and save packet to buffer
def receive_packet(received_buffer):
    data = request.data
    data = pickle.loads(data)
    received_buffer.append(data)
    return jsonify({'status': 'Data received'})


# Pause buffer's packet data, and save to data pool
def parse_received_packet(received_buffer, data_pool, ins_name):
    while len(received_buffer) > 0:
        curr_packet = received_buffer.popleft()
        # curr_packet = json.loads(curr_packet)

        # If packet TTL is 0, then not parse packet
        if getattr(curr_packet, 'TTL') == 0:
            continue

        temp_path = getattr(curr_packet, 'gossip_path')
        temp_path.append(ins_name)
        setattr(curr_packet, 'gossip_path', temp_path)

        temp_TTL = getattr(curr_packet, 'TTL') - 1
        setattr(curr_packet, 'TTL', temp_TTL)

        # If self is receiving data from this instance for the first time
        if getattr(curr_packet, 'src_ins') not in data_pool:
            new_Data = FedApp_Data()
            setattr(new_Data, getattr(curr_packet, 'data_type'), curr_packet)
            data_pool[getattr(curr_packet, 'src_ins')] = new_Data

        # If self has previously received data from this instance, but there is no data of this type
        elif getattr(curr_packet, 'src_ins') in data_pool and getattr(data_pool[getattr(curr_packet, 'src_ins')], getattr(curr_packet, 'data_type')) == None:
            setattr(data_pool[getattr(curr_packet, 'src_ins')], getattr(curr_packet, 'data_type'), curr_packet)

        # If self has previously received data from this instance, and there is data of this type, then compare packet time
        else:
            # previous_packet_time = getattr(getattr(data_pool[curr_packet['src_ins']], curr_packet['data_type']), 'time')
            previous_packet = getattr(data_pool[getattr(curr_packet, 'src_ins')], getattr(curr_packet, 'data_type'))
            previous_packet_time = getattr(previous_packet, 'time')
            current_packet_time = getattr(curr_packet, 'time')

            if current_packet_time > previous_packet_time:
                setattr(data_pool[getattr(curr_packet, 'src_ins')], getattr(curr_packet, 'data_type'), curr_packet)

    return received_buffer, data_pool
