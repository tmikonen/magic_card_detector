import json
import logging
import time

from django.core.management.base import BaseCommand, CommandError

from ..utils import get_card_bulk_data, get_set_data, map_ids_to_data, parse_ids_to_create_and_update
from ...models import Card, CardFace, CardSet

logger = logging.getLogger(__name__)


# V1: basic update and create time for 76,116 cards
class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and updates the Card Database'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int)

    def handle(self, *args, **options):
        t1 = time.time()
        all_cards_json = get_card_bulk_data()
        all_sets_json = get_set_data()
        t2 = time.time()

        total_cards = len(all_cards_json)
        logger.info(f'{total_cards:,} cards found')

        # finding out which sets to update and which sets to create
        set_ids_to_json, imported_set_ids = map_ids_to_data(all_sets_json)
        set_ids_to_create, set_ids_to_update = parse_ids_to_create_and_update(CardSet, imported_set_ids)

        # finding out which cards to update and which cards to create
        card_ids_to_json, imported_card_ids = map_ids_to_data(all_cards_json)
        card_ids_to_create, card_ids_to_update = parse_ids_to_create_and_update(Card, imported_card_ids)

        # bulk update or create sets


        # bulk update or create card faces


        # bulk update or create cards

        num = 1
        logger.info(f'Creating or updating {total_cards:,} cards')
        justify_width = len(f'{total_cards:,}')
        fail = "\033[91mFAILED"
        create = "\033[92mCREATED"
        update = "\033[96mUPDATED"

        abulk_create_cards(card_ids_to_json, ids_to_create)
        abulk_update_cards(card_ids_to_json, ids_to_update)

        t3 = time.time()
        logger.info(f'time to update all cards {(t3 - t2) // 60} mins {(t3 - t2) % 60} secs')

        t4 = time.time()
        logger.info(f'Time to get and update all cards {(t4 - t1) // 60} mins {(t4 - t1) % 60} secs')


def abulk_create_cards(card_ids_to_json, ids_to_create):
    objects = [
        Card(
            **Card.get_raw_json_for_bulk_operations(card_ids_to_json[card_uuid])
        ) for card_uuid in ids_to_create
    ]
    Card.objects.abulk_create(objects)


def abulk_update_cards(card_ids_to_json, ids_to_update):
    fields_to_update = []
    objects = [
        Card(
            **Card.get_raw_json_for_bulk_operations(card_ids_to_json[card_uuid])
        ) for card_uuid in ids_to_update
    ]
    Card.objects.abulk_update(objects, fields_to_update)

