import json
import logging
import time

from django.core.management.base import BaseCommand, CommandError

from ..utils import get_card_bulk_data
from ...models import Card, CardFace

logger = logging.getLogger(__name__)


# V1: basic update and create time for 76,116 cards
class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and updates the Card Database'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int)

    def handle(self, *args, **options):
        t1 = time.time()
        all_cards_json = get_card_bulk_data()
        t2 = time.time()

        total_cards = len(all_cards_json)
        logger.info(f'{total_cards:,} cards found')

        # finding out which cards to update and which cards to create
        card_ids_to_json = {j['id']: j for j in all_cards_json}
        imported_ids = set(card_ids_to_json.keys())
        current_ids = set(str(c['id']) for c in Card.objects.values('id'))

        ids_to_create = list(imported_ids.difference(current_ids))
        ids_to_update = list(imported_ids.difference(ids_to_create))

        # bulk update or create card faces


        # bulk update or create cards

        num = 1
        logger.info(f'Creating or updating {total_cards:,} cards')
        justify_width = len(f'{total_cards:,}')
        fail = "\033[91mFAILED"
        create = "\033[92mCREATED"
        update = "\033[96mUPDATED"

        abulk_create(card_ids_to_json, ids_to_create)
        abulk_update(card_ids_to_json, ids_to_update)

        t3 = time.time()
        logger.info(f'time to update all cards {(t3 - t2) // 60} mins {(t3 - t2) % 60} secs')

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

