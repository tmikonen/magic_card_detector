from django import forms

from .const import PRINTING_TYPE_OPTIONS
from.models import CardOwnership


class CreateCardOwnershipForm(forms.ModelForm):
    def __init__(self):
        super(CreateCardOwnershipForm, self).__init__()
        
    class Meta:
        model = CardOwnership
        fields = ['printing_type', 'price_purchased']
    # user_id = forms.IntegerField(label='User ID')
    # card_id = forms.UUIDField(label='Card ID')
    # printing_type = forms.Select(choices=PRINTING_TYPE_OPTIONS)
    # price_purchased = forms.DecimalField(label='Purchased Price', required=False)


