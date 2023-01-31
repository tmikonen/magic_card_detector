import datetime
import re
import uuid
import logging

from django.contrib.auth.models import User
from django.core.exceptions import FieldError
from django.db import models
from django.db.models import Count, F

from .const import CARD_LAYOUT_OPTIONS, PRINTING_TYPE_OPTIONS

logger = logging.getLogger(__name__)


#####    Non-card related   #####
# class User(models.Model):
#     """Represents a user of the system
#     """
#     id = models.UUIDField(primary_key=True, editable=False, default=uuid.uuid4)
#     first_name = models.CharField(max_length=256)
#     last_name = models.CharField(max_length=256)
#
#     # TODO: Replace with google OAuth
#     username = models.CharField(max_length=64)
#
#     def get_card_library_query_set(self):
#         return Card.objects.filter(cardownership__user__id=self.id)
#
#     def get_unique_card_library_query_set(self):
#         return self.get_card_library_query_set().distinct()
#
#     def get_unique_card_library_count_query_set(self):
#         return self.get_card_library_query_set().annotate(count=Count('unique_string'))


#########      Storage    ##########
class StorageLocation(models.Model):
    """Model for tracking where cards are stored
    """
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=1000)


##########      Cards     ############
class CardSet(models.Model):
    """The set that a card is a part of
    """
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=500)
    symbol = models.CharField(max_length=8)
    set_type = models.CharField(max_length=20)
    scryfall_uri = models.URLField()
    scryfall_set_cards_uri = models.URLField()
    icon_uri = models.URLField(null=True)

    @staticmethod
    def get_raw_json_for_bulk_operations(set_json):
        return {
            'id': set_json['id'],
            'name': set_json['name'],
            'symbol': set_json['code'],
            'set_type': set_json['set_type'],
            'scryfall_uri': set_json['scryfall_uri'],
            'scryfall_set_cards_uri': set_json['search_uri'],
            'icon_uri': set_json['icon_svg_uri'],
        }

    @staticmethod
    def non_id_fields():
        return [
            'name',
            'symbol',
            'set_type',
            'scryfall_uri',
            'scryfall_set_cards_uri',
            'icon_uri',
        ]


class Card(models.Model):
    """Represents a unique card in Magic's printing
    """
    id = models.UUIDField(primary_key=True)
    scryfall_uri = models.URLField()
    scryfall_url = models.URLField()

    layout = models.CharField(max_length=20, choices=CARD_LAYOUT_OPTIONS)
    name = models.CharField(max_length=500)

    conv_mana_cost = models.IntegerField()

    released_at = models.DateField(default=datetime.date(year=1993, month=9, day=1))
    time_added_to_db = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    # Foreign Relations
    card_set = models.ForeignKey(CardSet, on_delete=models.DO_NOTHING)

    def __str__(self):
        return f'{self.name} - {self.card_set.name}'

    @property
    def unique_string(self):
        return str(self.id)

    @staticmethod
    def non_id_fields():
        return [
            'scryfall_uri',
            'scryfall_url',
            'layout',
            'name',
            'released_at',
            'conv_mana_cost',
            'card_set_id',
        ]

    @staticmethod
    def get_raw_json_for_bulk_operations(card_json):
        return {
            'id': card_json['id'],
            'scryfall_uri': card_json['uri'],
            'scryfall_url': card_json['scryfall_uri'],
            'layout': card_json['layout'],
            'name': card_json['name'],
            'released_at': card_json['released_at'],
            'conv_mana_cost': int(card_json.get('cmc', 0)),
            'card_set_id': card_json['set_id'],
        }

    @classmethod
    def get_or_create_from_scryfall_json(cls, card_json):
        args_dict = cls.get_raw_json_for_bulk_operations(card_json)
        return cls.objects.get_or_create(**args_dict)

    @classmethod
    def update_or_create_from_scryfall_json(cls, card_json):
        args_dict = cls.get_raw_json_for_bulk_operations(card_json)
        return cls.objects.update_or_create(**args_dict)


class CardFace(models.Model):
    """Sometimes MTG cards can have more than one face tied to it. This represents that data
    """
    name = models.CharField(max_length=256)
    mana_cost = models.JSONField()

    power = models.SmallIntegerField(null=True)
    toughness = models.SmallIntegerField(null=True)

    type_line = models.CharField(max_length=256, null=True)
    oracle_text = models.CharField(max_length=1500, null=True)
    small_img_uri = models.URLField(null=True)
    normal_img_uri = models.URLField(null=True)

    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)

    @staticmethod
    def non_id_fields():
        return [
            'name',
            'mana_cost',
            'power',
            'toughness',
            'type_line',
            'oracle_text',
            'small_img_uri',
            'normal_img_uri',
            'card',
        ]

    @staticmethod
    def _parse_scryfall_json_to_model_args(card_json, card_id):
        toughness = card_json.get('toughness')
        if toughness:
            try:
                toughness = int(toughness)
            except ValueError:
                toughness = -1

        power = card_json.get('power')
        if power:
            try:
                power = int(power)
            except ValueError:
                power = -1

        return {
            'name': card_json['name'],
            'card_id': card_id,
            'mana_cost': CardFace._parse_mana_costs_from_scryfall_json(card_json['mana_cost']),
            'power': power,
            'toughness': toughness,
            'type_line': card_json.get('type_line'),
            'oracle_text': card_json.get('oracle_text'),
            'small_img_uri': card_json.get('image_uris').get('small'),
            'normal_img_uri': card_json.get('image_uris').get('normal'),
        }

    @staticmethod
    def _parse_mana_costs_from_scryfall_json(mana_json_string):
        mana = {}

        mana_regex = r'\{[1-9A-Z]\}'
        alt_mana_regex = r'\{B/P\}'
        mana_breakdown = re.findall(mana_regex, mana_json_string)
        alt_mana_breakdown = re.findall(alt_mana_regex, mana_json_string)

        for mana_sym in mana_breakdown:
            symbol = mana_sym.replace('{', "").replace("}", "")
            try:
                un_col = int(symbol)
                mana['un_col'] = un_col
            except ValueError:
                num = mana.get(symbol) or 0
                mana[symbol] = num + 1

        return {
            'green': mana.get('G'),
            'red': mana.get('R'),
            'blue': mana.get('U'),
            'black': mana.get('B'),
            'white': mana.get('W'),
            'colourless': mana.get('un_col'),
            'x_mana': mana.get('x') is not None,
            'other': len(alt_mana_breakdown) > 0,
        }

    @staticmethod
    def get_raw_json_for_bulk_operations(card_json):
        card_faces = card_json.get('card_faces')
        card_id = card_json['id']
        card_face_args = []
        imgs_on_card = card_json.get('image_uris', {})

        if card_faces:
            for face in card_faces:
                if not face.get('image_uris'):
                    face['image_uris'] = imgs_on_card

                card_face_args.append(CardFace._parse_scryfall_json_to_model_args(face, card_id))
        else:
            mana_cost = CardFace._parse_mana_costs_from_scryfall_json(card_json['mana_cost'])
            toughness = card_json.get('toughness')
            if toughness:
                try:
                    toughness = int(toughness)
                except ValueError:
                    toughness = -1

            power = card_json.get('power')
            if power:
                try:
                    power = int(power)
                except ValueError:
                    power = -1

            card_face_args.append(
                {
                    'name': card_json['name'],
                    'mana_cost': mana_cost,
                    'card_id': card_id,
                    'power': power,
                    'toughness': toughness,
                    'type_line': card_json['type_line'],
                    'oracle_text': card_json['oracle_text'],
                    'small_img_uri': card_json['image_uris']['small'],
                    'normal_img_uri': card_json['image_uris']['normal'],
                }
            )

        return card_face_args

    @classmethod
    def get_or_create_from_scryfall_json(cls, card_json):
        args_dict_list = cls.get_raw_json_for_bulk_operations(card_json)
        return [cls.objects.get_or_create(**args_dict) for args_dict in args_dict_list]

    @classmethod
    def update_or_create_from_scryfall_json(cls, card_json):
        args_dict_list = cls.get_raw_json_for_bulk_operations(card_json)
        return [cls.objects.update_or_create(**args_dict) for args_dict in args_dict_list]


class CardOwnership(models.Model):
    """Represents and instance of a card in a user's forms. So a user can have multiple copies of the same card
    """
    # Foreign Relations
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)
    printing_type = models.CharField(max_length=10, choices=PRINTING_TYPE_OPTIONS, default='normal')
    date_added = models.DateField(auto_now_add=True)
    date_removed = models.DateField(null=True)
    price_purchased = models.DecimalField(decimal_places=2, max_digits=9)
    price_sold = models.DecimalField(decimal_places=2, max_digits=9, null=True)


class CardPrice(models.Model):
    """Tracks how much a card cost in USD on a certain day
    """
    date = models.DateField(auto_now_add=True)
    price_usd = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    price_usd_foil = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    price_usd_etched = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    price_tix = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    price_eur = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    price_eur_foil = models.DecimalField(decimal_places=2, max_digits=9, null=True)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)

    @staticmethod
    def get_raw_json_for_bulk_operations(card_json, date_string):
        price_json = card_json['prices']
        return \
            {
                'date': date_string,
                'card_id': card_json['id'],
                'price_usd': price_json['usd'],
                'price_usd_foil': price_json['usd_foil'],
                'price_usd_etched': price_json['usd_etched'],
                'price_tix': price_json['tix'],
                'price_eur': price_json['eur'],
                'price_eur_foil': price_json['eur_foil']
            }


########      Decks     ############
class Deck(models.Model):
    """Represents a constructed Deck
    """
    name = models.CharField(max_length=50)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    cards = models.ManyToManyField(Card)
    current_location = models.ForeignKey(StorageLocation, on_delete=models.DO_NOTHING, null=True)

    def is_valid(self):
        raise NotImplemented


class ConstructedDeck(Deck):
    """Represents a deck for a 60 card constructed format
    """
    format_name = models.CharField(max_length=15, choices=[
        ('Standard', 'standard'),
        ('Modern', 'modern'),
        ('Pioneer', 'pioneer'),
        ('Historic', 'historic'),
        ('Legacy', 'legacy'),
        ('Vintage', 'vintage'),
        ('Pauper', 'pauper'),
    ])
    sideboard = models.ManyToManyField(Card)


class CommanderDeck(Deck):
    """Represents a deck for the commander format
    """
    commander = models.ForeignKey(Card, on_delete=models.DO_NOTHING, related_name='commander')
    secondary_commander = models.ForeignKey(Card, on_delete=models.DO_NOTHING, null=True,
                                            related_name='secondary_commander')
