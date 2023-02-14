import logging

from django.core.management.base import BaseCommand

from ..utils import \
    get_card_bulk_data, \
    get_set_data, \
    map_ids_to_data, \
    parse_ids_to_create_and_update, \
    timer
from ...models import Card, CardFace, CardSet

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and creates new objects in the Card Database. ' \
           '\n\tIf --update-and-create is specified, card data will also be updated'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int)

        parser.add_argument('--update-and-create', type=bool)

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
            set_fields_to_update = CardSet.non_id_fields()
            logger.info(f'Found {len(set_ids_to_create):,} sets to create and {len(set_ids_to_update):,} sets to update')
        print(f"Time to parse set data: {t():.4f} secs")

        # finding out which cards to update and which cards to create
        logger.info(f'Mapping Cards to ids')
        with timer() as t:
            card_ids_to_json, imported_card_ids = map_ids_to_data(all_cards_json)
            card_ids_to_create, card_ids_to_update = parse_ids_to_create_and_update(Card, imported_card_ids)
            logger.info(f'Found {len(card_ids_to_create):,} cards to create and {len(card_ids_to_update):,} cards to update')
            card_fields_to_update = Card.non_id_fields()
        print(f"Time to parse card data: {t():.4f} secs")

        # parsing out card face data
        logger.info(f'Splitting Card Faces between updating and creating')
        with timer() as t:
            card_faces_to_update = []
            card_faces_to_create = []
            count = 0
            for card_data in all_cards_json:
                card_face_data = CardFace.get_raw_json_for_bulk_operations(card_data)

                if card_face_data[0]['card_id'] in card_ids_to_create:
                    card_faces_to_create += [CardFace(**d) for d in card_face_data]
                else:
                    card_faces_to_update += [CardFace(**d) for d in card_face_data]

                count += 1
                if count % 1000 == 0:
                    logger.info(f'Parsed faces from {count:,} / {total_cards:,} cards')

            logger.info(f'Parsed faces from {count:,} / {total_cards:,} cards')
            face_fields_to_update = CardFace.non_id_fields()
            logger.info(f'Found {len(card_faces_to_create):,} card faces to create and {len(card_faces_to_update):,} card faces to update')
        print(f"Time to parse card face data: {t():.4f} secs")

        batch_size = options.get('batch_size') or 10000

        # creating new objects
        # bulk create sets
        logger.info(f'Creating {len(set_ids_to_create):,} Card Sets in batches of {batch_size:,}')
        with timer() as t:
            general_bulk_create(
                CardSet,
                set_ids_to_json,
                set_ids_to_create,
                batch_size
            )
        print(f"Time to create Card Sets: {t():.4f} secs")

        # bulk create cards
        logger.info(f'Creating {len(card_ids_to_create):,} Cards in batches of {batch_size:,}')
        with timer() as t:
            general_bulk_create(
                Card,
                card_ids_to_json,
                card_ids_to_create,
                batch_size
            )
        print(f"Time to create Cards: {t():.4f} secs")

        # bulk create card faces
        logger.info(f'Creating {len(card_faces_to_create):,} Card Faces in batches of {batch_size:,}')
        with timer() as t:
            CardFace.objects.bulk_create(card_faces_to_create, batch_size=batch_size)
        print(f"Time to create Card Faces: {t():.4f} secs")

        if options.get('update_and_create'):
            # bulk update card sets
            logger.info(f'Updating {len(set_ids_to_update):,} Card Sets in batches of {batch_size:,}')
            with timer() as t:
                general_bulk_update(
                    CardSet,
                    set_ids_to_json,
                    set_ids_to_update,
                    set_fields_to_update,
                    batch_size
                )
            print(f"Time to update Card Sets: {t():.4f} secs")

            # bulk update cards
            logger.info(f'Updating {len(card_ids_to_update):,} Cards in batches of {batch_size:,}')
            with timer() as t:
                general_bulk_update(
                    Card,
                    card_ids_to_json,
                    card_ids_to_update,
                    card_fields_to_update,
                    batch_size
                )
            print(f"Time to update Cards: {t():.4f} secs")

            # bulk update card faces
            # logger.info(f'Updating {len(card_faces_to_update):,} Card Faces in batches of {batch_size:,}')
            # with timer() as t:
            #     CardFace.objects.bulk_update(card_faces_to_update, face_fields_to_update, batch_size=batch_size)
            # print(f"Time to update Card Faces: {t():.4f} secs")


def general_bulk_create(obj_class, ids_to_data_mapping, ids_to_create, batch_size):
    obj_class.objects.bulk_create(
        [
            obj_class(**obj_class.get_raw_json_for_bulk_operations(ids_to_data_mapping[create_id])) for
            create_id in ids_to_create
        ],
        batch_size=batch_size
    )


def general_bulk_update(obj_class, ids_to_data_mapping, ids_to_update, fields_to_update, batch_size):
    last_batch = 0

    for i in range(batch_size, len(ids_to_update), batch_size):
        logger.info(f'\tUpdating {obj_class} from index {last_batch} to {i-1}')
        obj_class.objects.bulk_update(
            [
                obj_class(**obj_class.get_raw_json_for_bulk_operations(ids_to_data_mapping[update_id])) for
                update_id in ids_to_update[last_batch: i]
            ],
            fields=fields_to_update,
            batch_size=batch_size,
        )
        last_batch = i


def general_bulk_update_or_create(obj_class, ids_to_data_mapping, ids_to_update, ids_to_create, fields_to_update, batch_size):
    general_bulk_update(obj_class, ids_to_data_mapping, ids_to_update, fields_to_update, batch_size)
    general_bulk_create(obj_class, ids_to_data_mapping, ids_to_create, batch_size)

