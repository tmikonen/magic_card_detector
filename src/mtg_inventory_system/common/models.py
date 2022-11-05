import uuid

from django.db import models
from django.db.models import Count, F

from const import CARD_LAYOUT_OPTIONS, PRINTING_TYPE_OPTIONS


class User(models.Model):
    id = models.UUIDField(primary_key=True, editable=False, default=uuid.uuid4)
    first_name = models.CharField(max_length=256)
    last_name = models.CharField(max_length=256)

    def get_card_library_query_set(self):
        return Card.objects.filter(cardownership__user__id=self.id)

    def get_unique_card_library_query_set(self):
        return self.get_card_library_query_set().distinct()

    def get_unique_card_library_count_query_set(self):
        return self.get_card_library_query_set().annotate(count=Count('unique_string'))


class Set(models.Model):
    name = models.CharField(max_length=500)
    scryfall_uuid = models.UUIDField()


class ManaCost(models):
    green = models.PositiveSmallIntegerField(null=True, default=None)
    red = models.PositiveSmallIntegerField(null=True, default=None)
    blue = models.PositiveSmallIntegerField(null=True, default=None)
    black = models.PositiveSmallIntegerField(null=True, default=None)
    white = models.PositiveSmallIntegerField(null=True, default=None)
    colourless = models.PositiveSmallIntegerField(null=True, default=None)
    x_mana = models.BooleanField(default=False)

    class Meta:
        unique_together = ('green', 'red', 'blue', 'black', 'white', 'colourless')


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


class Card(models.Model):
    uuid = models.UUIDField(primary_key=True, editable=False)
    scryfall_uri = models.URLField()
    scryfall_url = models.URLField()

    layout = models.CharField(max_length=20, choices=CARD_LAYOUT_OPTIONS)
    name = models.CharField(max_length=500)

    conv_mana_cost = models.PositiveSmallIntegerField()
    printing_type = models.CharField(max_length=10, choices=PRINTING_TYPE_OPTIONS)

    # Foreign Relations
    face_primary = models.OneToOneField(CardFace, on_delete=models.CASCADE)
    face_secondary = models.OneToOneField(CardFace, on_delete=models.CASCADE, null=True)
    set = models.ForeignKey(Set, on_delete=models.DO_NOTHING)

    class Meta:
        unique_together = ('uuid', 'printing_type')

    @property
    def unique_string(self):
        return "{} {}".format(self.uuid, self.printing_type)


class CardOwnership(models.Model):
    # Foreign Relations
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)
    date_added = models.DateField(auto_now_add=True)
    date_removed = models.DateField(null=True)
    price_purchased = models.DecimalField(decimal_places=2)
    price_sold = models.DecimalField(decimal_places=2, null=True)


class CardPrice(models.Model):
    """Tracks how much a card cost in USD on a certain day
    """
    date = models.DateField(auto_now_add=True)
    price_usd = models.DecimalField(decimal_places=2)
    card = models.OneToOneField(Card, on_delete=models.DO_NOTHING)

