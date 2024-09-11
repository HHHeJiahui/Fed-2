import time
import json

def get_packet(src_ins, dest_ins, data, data_type):
    return json.dumps({
        'src_ins': src_ins,
        'dest_ins': dest_ins,
        'time': time.time(),
        'TTL': 3,
        'data': data,
        'data_type': data_type,
        'gossip_path': []
    })