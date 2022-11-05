import uuid

from django.db import models


class User(models.Model):
    id = models.UUIDField(primary_key=True, editable=False, default=uuid.uuid4)
    first_name = models.CharField(max_length=256)
    last_name = models.CharField(max_length=256)


class Set(models.Model):
    name = models.CharField(max_length=500)
    scryfall_uuid = models.UUIDField()


class CardFace(models.Model):
    """Sometimes MTG cards can have more than one face tied to it. This represents that data
    """
    name
    mana
    types
    text
    colours
    small_img_uri
    normal_img_uri


class Card(models.Model):
    uuid = models.UUIDField(primary_key=True, editable=False)
    scryfall_uri = models.CharField(max_length=500)

    is_two_sided = models.BooleanField(default=False)

    # Primary Section
    name = models.CharField(max_length=500)
    power = models.IntegerField()
    toughness = models.IntegerField()

    small_img_url = models.CharField(max_length=500)
    normal_img_url = models.CharField(max_length=500)

    # Secondary Section

    small_img_url_secondary = models.CharField(max_length=500, null=True)
    normal_img_url_back = models.CharField(max_length=500, null=True)

    # Foreign Relations
    set = models.ForeignKey(Set, on_delete=models.DO_NOTHING)

class CardOwnership(models.Model):
    # Foreign Relations
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    card = models.ForeignKey(Card, on_delete=models.DO_NOTHING)

