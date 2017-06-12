import json
import time

import requests

with open('token.json') as f:
    apikey = json.load(f)

headers = {
    'Authorization': 'Bearer ' + apikey['oauth2']['access_token']
}

freesound_download_dir = 'data/freesound_whooshes'
text_search_url = 'http://www.freesound.org/apiv2/search/text/'
sound_download_url_tpl = 'https://www.freesound.org/apiv2/sounds/{}/download/'


def search_tag(tag):
    params = {
        'filter': tag,
        'token': apikey['api']['secret']
    }
    sounds = []
    try:
        r = requests.get(text_search_url, params)
        sounds.extend([s['id'] for s in r.json()['results']])
        while r.json()['next'] is not None:
            r = requests.get(r.json()['next'], {'token': params['token']})
            sounds.extend([s['id'] for s in r.json()['results']])
            time.sleep(0.5)
    finally:
        with open(freesound_download_dir + '/sounds.json', 'w') as f:
            json.dump(sounds, f)


def download_sound(sid):
    r = requests.get(sound_download_url_tpl.format(sid), headers=headers)
    r.raw.decode_content = True
    filetype = r.headers['Content-Disposition'].split('.')[-1].replace('"', '')
    with open('{}/{}.{}'.format(freesound_download_dir, sid, filetype), 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    with open('data/freesound_whoosh_list.txt', 'r') as f:
        sids = [int(x) for x in f.readlines()]
        for i, s in enumerate(sids):
            print('{}\t{}'.format(i, s))
            download_sound(s)
            time.sleep(1)
