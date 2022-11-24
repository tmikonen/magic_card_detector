import json
import logging
import time

import requests
from django.core.paginator import Paginator

from django.db.models import F, Q
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from urllib.parse import urlencode

from .models import Card

logger = logging.getLogger(__name__)


def index(req):
    return HttpResponse("Hello, world. You're at the common index.")


def cards(req):
    all_cards = Card.objects.all().annotate(set_name=F('card_set__name')).order_by('name')
    card_paginator = Paginator(all_cards, 25)

    page = req.GET.get('page') or 1
    page_obj = card_paginator.get_page(page)

    return render(req, 'cards/list.html', {'page_obj': page_obj})


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
        Card.get_or_create_from_scryfall_json(card_data)

    return HttpResponseRedirect(reverse(cards))
