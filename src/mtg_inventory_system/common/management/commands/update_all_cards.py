import json
import logging
import time

import requests
from django.core.management.base import BaseCommand, CommandError

from ...models import Card

logger = logging.getLogger(__name__)


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
                    logger.info(f'{len(all_cards_json):,} cards found')
                    logger.debug(json.dumps(all_cards_json[0:4], indent=3))

                    num = 0
                    max_cards = options.get('max_cards', len(all_cards_json))
                    for card_json in all_cards_json:

                        if num < max_cards:
                            try:
                                card_obj, created = Card.update_or_create_from_scryfall_json(card_json)
                                logger.info(f'{num} {"Created" if created else "Updated"} card {str(card_obj)}')
                            except Exception as e:
                                logger.error(f'{num} Failed to create card {card_json["name"]} because {str(e)}')
                            num += 1
                        else:
                            break

                    t3 = time.time()
                    logger.info(f'time to update all cards {(t3 - t2) // 60} mins {(t3 - t2) % 60} secs')
            else:
                logger.warning('Could not find a default card uri. Cannot update dataset')
        else:
            logger.error(f'Status: {bulk_data_location_request.status_code} for request {request_url}')

        t4 = time.time()
        logger.info(f'Time to get and update all cards {(t4 - t1) // 60} mins {(t4 - t1) % 60} secs')
