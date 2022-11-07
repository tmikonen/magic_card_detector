from django.contrib import admin

# Register your models here.
from .models import *

admin.register(User)
admin.register(StorageLocation)
admin.register(CardSet)
admin.register(ManaCost)
admin.register(CardFace)
admin.register(Card)
admin.register(CardOwnership)
admin.register(CardPrice)
admin.register(Deck)
admin.register(ConstructedDeck)
admin.register(CommanderDeck)
