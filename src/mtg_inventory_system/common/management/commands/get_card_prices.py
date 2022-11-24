import datetime
import logging

from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.core.management.base import BaseCommand

from ..utils import \
    get_card_bulk_data, \
    get_set_data, \
    map_ids_to_data, \
    parse_ids_to_create_and_update, \
    timer
from ...models import Card, CardFace, CardSet, CardPrice

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Gets bulk data from Scryfall and creates new objects in the Card Database. ' \
           '\n\tIf --update-and-create is specified, card data will also be updated'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int)

    def handle(self, *args, **options):
        today = datetime.date.today()
        today_str = today.strftime('%y-%m-%d')

        # find if we have already got the price data for today
        try:
            CardPrice.objects.get(date=today)
        except MultipleObjectsReturned:
            logger.info('Already have card price data for today')
            return
        except ObjectDoesNotExist:
            pass

        with timer() as t:
            all_cards_json = get_card_bulk_data()
        print(f"Time to get remote data: {t():.4f} secs")

        batch_size = options.get('batch_size') or 10000
        total_cards = len(all_cards_json)
        logger.info(f'{total_cards:,} cards found')

        # Parsing out the card price data
        logger.info(f'Parsing out the card price data')
        with timer() as t:
            card_prices = [
                CardPrice.get_raw_json_for_bulk_operations(card_json, date_string=today_str)
                for card_json in all_cards_json
            ]
        print(f"Time to parse {len(card_prices):,} card prices: {t():.4f} secs")

        # creating the card prices
        logger.info(f'Creating {len(card_prices):,} Card Prices in batches of {batch_size:,}')
        with timer() as t:
            CardPrice.objects.bulk_create(
                [CardPrice(**card_price) for card_price in card_prices],
                batch_size=batch_size
            )
        print(f"Time to create Card Faces: {t():.4f} secs")

