from flask import jsonify
import requests


def get_peers(temp_ins):
    try:
        response = requests.get(f'https://{temp_ins}/api/v1/instance/peers', params={'local': True})
        peers = response.json()
        return peers
    except:
        peers = None
        return peers


def get_hashtags(temp_ins):
    try:
        limit = 20
        offset = 0
        hashtags = []

        for i in range(5):
            params = {
                'limit': limit,
                'offset': offset,
                'local': True
            }
            response = requests.get(f'https://{temp_ins}/api/v1/trends/tags', params=params)
            if response.status_code == 200:
                trends = response.json()
                if not trends:
                    break
                for trend in trends:
                    hashtags.append(trend['name'])
                offset += limit
            else:
                print("Failed to retrieve trending tags")
                break
        return hashtags
    except:
        hashtags = None
        return hashtags


# Token: 2E60GOkLPWdPEwy14KxrrYuaIGy-KEhb8mrrYC9EAls
def get_blocks(temp_ins):
    try:
        access_token = '2E60GOkLPWdPEwy14KxrrYuaIGy-KEhb8mrrYC9EAls'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(f'https://{temp_ins}/api/v1/admin/domain_blocks', headers=headers, params={'local': True})
        blocks = response.json()
        blocks = [item['domain'] for item in blocks]
        return blocks
    except:
        blocks = None
        return blocks

def get_rules(temp_ins):
    try:
        response = requests.get(f'https://{temp_ins}/api/v1/instance/rules', params={'local': True})
        rules = response.json()
        rules = [item['text'] for item in rules]
        return rules
    except:
        rules = None
        return rules

def get_all(temp_ins):
    peers = get_peers(temp_ins)
    hashtags = get_hashtags(temp_ins)
    blocks = get_blocks(temp_ins)
    rules = get_rules(temp_ins)

    return {
        'peers': peers,
        'hashtags': hashtags,
        'block_list': blocks,
        'rules': rules
    }
