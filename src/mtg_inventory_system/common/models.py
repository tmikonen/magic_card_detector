import re
import uuid
import logging

from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import FieldError
from django.db import models
from django.db.models import Count, F

from .const import CARD_LAYOUT_OPTIONS, PRINTING_TYPE_OPTIONS

logger = logging.getLogger(__name__)


#####    Non-card related   #####
class User(models.Model):
    """Represents a user of the system
    """
    id = models.UUIDField(primary_key=True, editable=False, default=uuid.uuid4)
    first_name = models.CharField(max_length=256)
    last_name = models.CharField(max_length=256)

    # TODO: Replace with google OAuth
    username = models.CharField(max_length=64)

    def get_card_library_query_set(self):
        return Card.objects.filter(cardownership__user__id=self.id)

    def get_unique_card_library_query_set(self):
        return self.get_card_library_query_set().distinct()

    def get_unique_card_library_count_query_set(self):
        return self.get_card_library_query_set().annotate(count=Count('unique_string'))


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
    scryfall_uuid = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=500)
    symbol = models.CharField(max_length=8)
    scryfall_uri = models.URLField()
    scryfall_set_cards_uri = models.URLField()


class ManaCost(models.Model):
    green = models.PositiveSmallIntegerField(null=True, default=None)
    red = models.PositiveSmallIntegerField(null=True, default=None)
    blue = models.PositiveSmallIntegerField(null=True, default=None)
    black = models.PositiveSmallIntegerField(null=True, default=None)
    white = models.PositiveSmallIntegerField(null=True, default=None)
    colourless = models.PositiveSmallIntegerField(null=True, default=None)
    x_mana = models.BooleanField(default=False)
    other = models.BooleanField(default=False)

    class Meta:
        unique_together = ('green', 'red', 'blue', 'black', 'white', 'colourless')

    @classmethod
    def from_scryfall_json(cls, mana_json_string):
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

        return cls.objects.get_or_create(
            green=mana.get('G'),
            red=mana.get('R'),
            blue=mana.get('U'),
            black=mana.get('B'),
            white=mana.get('W'),
            colourless=mana.get('un_col'),
            x_mana=mana.get('x') is not None,
            other=len(alt_mana_breakdown) > 0,
        )[0]


class CardFace(models.Model):
    """Sometimes MTG cards can have more than one face tied to it. This represents that data
    """
    name = models.CharField(max_length=128)
    mana_cost = models.ForeignKey(ManaCost, on_delete=models.DO_NOTHING)

    power = models.SmallIntegerField(null=True)
    toughness = models.SmallIntegerField(null=True)

    type_line = models.CharField(max_length=100)
    oracle_text = models.CharField(max_length=500)
    small_img_uri = models.URLField()
    normal_img_uri = models.URLField()

    @classmethod
    def from_scryfall_json(cls, card_face_json):
        return cls.objects.get_or_create(
            name=card_face_json['name'],
            mana_cost=ManaCost.from_scryfall_json(card_face_json['mana_cost']),
            power=card_face_json.get('power'),
            toughness=card_face_json.get('toughness'),
            type_line=card_face_json['type_line'],
            oracle_text=card_face_json['oracle_text'],
            small_img_uri=card_face_json['image_uris']['small'],
            normal_img_uri=card_face_json['image_uris']['normal'],
        )[0]


class Card(models.Model):
    """Represents a unique card in Magic's printing
    """
    uuid = models.UUIDField(primary_key=True)
    scryfall_uri = models.URLField()
    scryfall_url = models.URLField()

    layout = models.CharField(max_length=20, choices=CARD_LAYOUT_OPTIONS)
    name = models.CharField(max_length=500)

    conv_mana_cost = models.PositiveSmallIntegerField()
    printing_type = models.CharField(max_length=10, choices=PRINTING_TYPE_OPTIONS, default='normal')

    # Foreign Relations
    face_primary = models.ForeignKey(CardFace, on_delete=models.CASCADE, related_name='face_primary')
    face_secondary = models.ForeignKey(CardFace, on_delete=models.CASCADE, null=True, related_name='face_secondary')
    card_set = models.ForeignKey(CardSet, on_delete=models.DO_NOTHING)

    class Meta:
        unique_together = ('uuid', 'printing_type')

    @property
    def unique_string(self):
        return "{} {}".format(self.uuid, self.printing_type)

    @classmethod
    def from_scryfall_json(cls, card_json):
        card_faces = card_json.get('card_faces')

        if card_faces and len(card_faces) == 2:
            if card_faces[0].get('image_uris'):
                primary_face = CardFace.from_scryfall_json(card_faces[0])
                secondary_face = CardFace.from_scryfall_json(card_faces[1])
            else:
                card_faces[0]['image_uris'] = card_json['image_uris']
                card_faces[1]['image_uris'] = card_json['image_uris']

                primary_face = CardFace.from_scryfall_json(card_faces[0])
                secondary_face = CardFace.from_scryfall_json(card_faces[1])
        elif card_faces and len(card_faces) > 2:
            error_msg = "Cannot create a card {} with {} faces / alternates. url: {}".format(card_json['name'],
                                                                                             len(card_faces),
                                                                                             card_json['url'])
            logger.error(error_msg)
            raise FieldError(error_msg)
        else:
            mana_cost = ManaCost.from_scryfall_json(card_json['mana_cost'])
            primary_face = CardFace.objects.get_or_create(
                name=card_json['name'],
                mana_cost=mana_cost,
                power=card_json.get('power'),
                toughness=card_json.get('power'),
                type_line=card_json['type_line'],
                oracle_text=card_json['oracle_text'],
                small_img_uri=card_json['image_uris']['small'],
                normal_img_uri=card_json['image_uris']['normal'],
            )[0]
            secondary_face = None

        card_set = CardSet.objects.get_or_create(
            scryfall_uuid=card_json['set_id'],
            name=card_json['set_name'],
            symbol=card_json['set'],
            scryfall_uri=card_json['set_uri'],
            scryfall_set_cards_uri=card_json['set_search_uri'],
        )[0]

        return cls.objects.get_or_create(
            uuid=card_json['id'],
            scryfall_uri=card_json['uri'],
            scryfall_url=card_json['scryfall_uri'],
            layout=card_json['layout'],
            name=card_json['name'],
            conv_mana_cost=int(card_json['cmc']),
            face_primary=primary_face,
            face_secondary=secondary_face,
            card_set=card_set,
        )[0]


class CardOwnership(models.Model):
    """Represents and instance of a card in a user's library. So a user can have multiple copies of the same card
    """
    # Foreign Relations
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)
    date_added = models.DateField(auto_now_add=True)
    date_removed = models.DateField(null=True)
    price_purchased = models.DecimalField(decimal_places=2, max_digits=9)
    price_sold = models.DecimalField(decimal_places=2, max_digits=9, null=True)


class CardPrice(models.Model):
    """Tracks how much a card cost in USD on a certain day
    """
    date = models.DateField(auto_now_add=True)
    price_usd = models.DecimalField(decimal_places=2, max_digits=9)
    price_tix = models.DecimalField(decimal_places=2, max_digits=9)
    prince_eur = models.DecimalField(decimal_places=2, max_digits=9)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)


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
