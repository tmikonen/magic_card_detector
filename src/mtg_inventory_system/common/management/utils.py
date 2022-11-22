import json
import logging

import requests

from time import perf_counter
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_card_bulk_data():
    request_url = 'https://api.scryfall.com/bulk-data'
    bulk_data_location_request = requests.get(
        request_url,
        headers={"Accept": "application/json"}
    )

    if bulk_data_location_request.status_code == 200:
        bulk_data_meta = json.loads(bulk_data_location_request.text)['data']

        default_cards_uri = None
        for bulk_data_type in bulk_data_meta:
            if bulk_data_type['type'] == 'default_cards':
                default_cards_uri = bulk_data_type['download_uri']
                logger.info(f'Got bulk card uri at {default_cards_uri}')
                break

        if default_cards_uri:
            bulk_card_request = requests.get(default_cards_uri)

            if bulk_card_request.status_code == 200:
                all_cards_json = json.loads(bulk_card_request.text)
                return all_cards_json

    return []


def get_set_data():
    request_url = 'https://api.scryfall.com/sets/'
    set_data_request = requests.get(
        request_url,
        headers={"Accept": "application/json"}
    )

    if set_data_request.status_code == 200:
        set_data_json = json.loads(set_data_request.text)['data']
        return set_data_json

    return []


def map_ids_to_data(json_data, id_name='id'):
    ids_to_json = {j[id_name]: j for j in json_data}
    imported_ids = set(ids_to_json.keys())

    return ids_to_json, imported_ids


def parse_ids_to_create_and_update(obj_class, imported_ids, id_name='id'):
    current_ids = set(str(c[id_name]) for c in obj_class.objects.values(id_name))

    ids_to_create = list(imported_ids.difference(current_ids))
    ids_to_update = list(imported_ids.difference(ids_to_create))

    return ids_to_create, ids_to_update


@contextmanager
def timer() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
