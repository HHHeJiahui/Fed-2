from peer_selector import select_peers
from send_receive_packet import send_packet, receive_packet, parse_received_packet
from packet_construction import get_packet
from FedApp_class import FedApp_Packet

def gossip_controller(instance_name, data, data_pool, selector, stat, n:int=0, frac:int=0, send_type:str=None):
    gossip_peers = select_peers(getattr(data, 'peers'), selector, n, frac)

    # 1. send self data with specific data type
    # 2. send data pool's data
    if len(gossip_peers) != 0:
        for peer in gossip_peers:
            curr_self_data = getattr(data, send_type)

            # 1. send self data
            if curr_self_data != None:
                # curr_size = get_packet_size(curr_self_data)
                packet = FedApp_Packet(src_ins=instance_name, dest_ins=peer, data=curr_self_data, data_type=send_type)

                #### Solve the data format problem ####
                # packet = json.loads(packet)
                # packet['gossip_path'].append(instance_name)
                send_packet(peer, packet)

            # 2. send data pool's data
            if len(data_pool) != 0:
                for need_gossip_ins in list(data_pool):
                    if need_gossip_ins == peer:
                        continue
                    # For each data type, if it's not empty, then send
                    for curr_data_type in list(vars(data_pool[need_gossip_ins]).keys()):
                        if getattr(data_pool[need_gossip_ins], curr_data_type) != None:
                            curr_gossip_packet = getattr(data_pool[need_gossip_ins], curr_data_type)
                            # curr_size = get_packet_size(curr_gossip_data)
                            send_packet(peer, curr_gossip_packet)
            # Update stats
            stat['gossip_count'] += 1
    return stat