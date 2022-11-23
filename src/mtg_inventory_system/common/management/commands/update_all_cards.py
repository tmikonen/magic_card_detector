import json
import logging
import time

from django.core.management.base import BaseCommand, CommandError

from ..utils import \
    get_card_bulk_data, \
    get_set_data, \
    map_ids_to_data, \
    parse_ids_to_create_and_update, \
    timer
from ...models import Card, CardFace, CardSet

logger = logging.getLogger(__name__)


# V1: basic update and create time for 76,116 cards
class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and updates the Card Database'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int)

    def handle(self, *args, **options):
        with timer() as t:
            all_cards_json = get_card_bulk_data()
            all_sets_json = get_set_data()
        print(f"Time to get remote data: {t():.4f} secs")

        total_cards = len(all_cards_json)
        total_sets = len(all_sets_json)
        logger.info(f'{total_cards:,} cards found')
        logger.info(f'{total_sets:,} sets found')

        # finding out which sets to update and which sets to create
        logger.info(f'Mapping Sets to ids')
        with timer() as t:
            set_ids_to_json, imported_set_ids = map_ids_to_data(all_sets_json)
            set_ids_to_create, set_ids_to_update = parse_ids_to_create_and_update(CardSet, imported_set_ids)
            logger.info(f'Found {len(set_ids_to_create)} sets to create and {len(set_ids_to_update)} sets to update')
        print(f"Time to parse set data: {t():.4f} secs")

        # finding out which cards to update and which cards to create
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            card_ids_to_json, imported_card_ids = map_ids_to_data(all_cards_json)
            card_ids_to_create, card_ids_to_update = parse_ids_to_create_and_update(Card, imported_card_ids)
            logger.info(f'Found {len(card_ids_to_create)} cards to create and {len(card_ids_to_update)} cards to update')
        print(f"Time to parse card data: {t():.4f} secs")

        # parsing out card face data
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            card_faces_to_update = []
            card_faces_to_create = []
            for card_data in all_cards_json:
                card_face_data = CardFace.get_raw_json_for_bulk_operations(card_data)

                if card_face_data['card_id'] in card_ids_to_create:
                    card_faces_to_create += card_face_data
                else:
                    card_faces_to_update += card_face_data

            logger.info(f'Found {len(card_faces_to_create)} cards to create and {len(card_faces_to_update)} cards to update')
        print(f"Time to parse card face data: {t():.4f} secs")

        # bulk update or create sets
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            general_bulk_update_or_create(
                CardSet,
                set_ids_to_json,
                set_ids_to_update,
                set_ids_to_create
            )
        print(f"Time to update or create Card Sets: {t():.4f} secs")

        # bulk update or create cards
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            general_bulk_update_or_create(
                Card,
                card_ids_to_json,
                card_ids_to_update,
                card_ids_to_create
            )
        print(f"Time to update or create Cards: {t():.4f} secs")

        # bulk update or create card faces
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            CardFace.objects.bulk_update(card_faces_to_update)
            CardFace.objects.bulk_create(card_faces_to_create)
        print(f"Time to update or create Card Faces: {t():.4f} secs")


def general_bulk_update_or_create(obj_class, ids_to_data_mapping, ids_to_update, ids_to_create):
    obj_class.objects.bulk_update([
        obj_class.get_raw_json_for_bulk_operations(ids_to_data_mapping[update_id]) for
        update_id in ids_to_update
    ])
    obj_class.objects.bulk_create([
        obj_class.get_raw_json_for_bulk_operations(ids_to_data_mapping[create_id]) for
        create_id in ids_to_create
    ])

