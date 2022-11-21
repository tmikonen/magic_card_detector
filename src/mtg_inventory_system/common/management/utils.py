import json
import logging

import requests

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
