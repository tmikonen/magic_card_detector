import json
import logging
import time

import requests
from django.core.management.base import BaseCommand, CommandError

from ...models import Card

logger = logging.getLogger(__name__)


# V1: basic update and create time for 76,116 cards
class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and updates the Card Database'

    def add_arguments(self, parser):
        parser.add_argument('--max-cards', type=int)

    def handle(self, *args, **options):
        t1 = time.time()
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
                t2 = time.time()
                logger.info(f"Time to get all cards {(t2 - t1) // 60} mins {(t2 - t1) % 60} secs")

                if bulk_card_request.status_code == 200:
                    all_cards_json = json.loads(bulk_card_request.text)
                    total_cards = len(all_cards_json)
                    logger.info(f'{total_cards:,} cards found')
                    logger.debug(json.dumps(all_cards_json[0:4], indent=3))

                    num = 1
                    max_cards = options.get('max_cards') or total_cards
                    logger.info(f'Creating or updating {max_cards:,} cards')
                    justify_width = len(f'{max_cards:,}')
                    fail = "\033[91mFAILED"
                    create = "\033[92mCREATED"
                    update = "\033[96mUPDATED"

                    card_ids_to_json = {j['id']: j for j in all_cards_json}
                    imported_ids = set(card_ids_to_json.keys())
                    current_ids = set(str(c['id']) for c in Card.objects.values('id'))

                    ids_to_create = list(imported_ids.difference(current_ids))
                    ids_to_update = list(imported_ids.difference(ids_to_create))

                    abulk_create(card_ids_to_json, ids_to_create)
                    abulk_update(card_ids_to_json, ids_to_update)

                    # for card_json in all_cards_json:
                    #     if num <= max_cards:
                    #         try:
                    #             card_obj, created = Card.update_or_create_from_scryfall_json(card_json)
                    #             logger.info(f'\033[1m{num: >{justify_width},}\033[0m of {max_cards:,} {create if created else update: <10} card {str(card_obj)}\033[0m')
                    #         except Exception as e:
                    #             err_str = str(e).replace("\n", "   |   ")
                    #             logger.error(f'\033[1m{num: >{justify_width},}\033[0m of {max_cards:,} {fail: <10} to create card {card_json["name"]} because {err_str}\033[0m')
                    #         num += 1
                    #     else:
                    #         break

                    t3 = time.time()
                    logger.info(f'time to update all cards {(t3 - t2) // 60} mins {(t3 - t2) % 60} secs')
            else:
                logger.warning('Could not find a default card uri. Cannot update dataset')
        else:
            logger.error(f'Status: {bulk_data_location_request.status_code} for request {request_url}')

        t4 = time.time()
        logger.info(f'Time to get and update all cards {(t4 - t1) // 60} mins {(t4 - t1) % 60} secs')


def abulk_create(card_ids_to_json, ids_to_create):
    objects = [
        Card(
            **Card.get_raw_json_for_bulk_operations(card_ids_to_json[card_uuid])
        ) for card_uuid in ids_to_create
    ]
    Card.objects.abulk_create(objects)


def abulk_update(card_ids_to_json, ids_to_update):
    fields_to_update = []
    objects = [
        Card(
            **Card.get_raw_json_for_bulk_operations(card_ids_to_json[card_uuid])
        ) for card_uuid in ids_to_update
    ]
    Card.objects.abulk_update(objects, fields_to_update)
