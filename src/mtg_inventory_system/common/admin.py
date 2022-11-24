from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(User)
admin.site.register(StorageLocation)
admin.site.register(CardSet)
admin.site.register(CardFace)
admin.site.register(Card)
admin.site.register(CardOwnership)
admin.site.register(CardPrice)
admin.site.register(Deck)
admin.site.register(ConstructedDeck)
admin.site.register(CommanderDeck)
