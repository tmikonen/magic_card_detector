from django.db.models import F, Q
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from .models import Card


def index(request):
    return HttpResponse("Hello, world. You're at the common index.")


def cards(request):
    all_cards = Card.objects.all().annotate(set_name=F('card_set__name'))
    template = loader.get_template('cards/index.html')
    context = {
        'all_cards': all_cards,
    }

    return HttpResponse(template.render(context, request))


def card(request, card_uuid):
    return HttpResponse("You're looking at card {}".format(card_uuid))
