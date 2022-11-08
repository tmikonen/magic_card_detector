import json
import logging
import requests

from django.db.models import F, Q
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from .models import Card

logger = logging.getLogger(__name__)


def index(req):
    return HttpResponse("Hello, world. You're at the common index.")


def cards(req):
    all_cards = Card.objects.all().annotate(set_name=F('card_set__name'))
    template = loader.get_template('cards/index.html')
    context = {
        'all_cards': all_cards,
    }

    return HttpResponse(template.render(context, req))


def card(req, card_uuid):
    return HttpResponse("You're looking at card {}".format(card_uuid))


def import_cards_from_url(req):
    template = loader.get_template('cards/import_from_request.html')
    context = {

    }
    return HttpResponse(template.render(context, req))


def run_import(req):
    url = req.POST['scryfall_url']
    logger.info(url)

    response = requests.get(url, headers={"Accept": "application/json"})
    json_data = json.loads(response.text)

    if json_data['object'] == 'card':
        card_json = [json_data]
    elif json_data['object'] == 'list':
        card_json = json_data['data']
    else:
        return HttpResponseRedirect(reverse(cards))

    for card_data in card_json:
        Card.from_scryfall_json(card_data)

    return HttpResponseRedirect(reverse(cards))



