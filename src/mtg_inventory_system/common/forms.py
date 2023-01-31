from django import forms

from .const import PRINTING_TYPE_OPTIONS
from .models import CardOwnership


class CreateCardOwnershipForm(forms.Form):
    print_type = forms.ChoiceField(required=True, choices=PRINTING_TYPE_OPTIONS)
    price_purchased = forms.DecimalField(required=True, initial=0.0)
    num_of_cards = forms.IntegerField(required=True, initial=1)

    def add_to_library(self, user):
        vals = self.cleaned_data
        for i in range(vals['num_of_cards']):
            card_own = CardOwnership(
                user=user,
                card=self.card,
                printing_type=vals['print_type'],
                price_purchased=vals['price_purchased']
            )
            card_own.save()


