import json
import logging
import time

import requests
from django.conf import settings
from django.contrib.auth.models import User
from django.core.paginator import Paginator

from django.db.models import F, Q, Value, URLField, Subquery, OuterRef
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from urllib.parse import urlencode

from .models import Card, CardFace, CardOwnership

logger = logging.getLogger(__name__)


def index(req):
    return HttpResponse("Hello, world. You're at the common index.")


def cards(req):
    all_cards = Card.objects.all().annotate(set_name=F('card_set__name')).order_by('name').annotate(
        card_img=Subquery(
            CardFace.objects.filter(
                card__id=OuterRef('id')
            ).distinct('card__id').values('small_img_uri')

        )
    )
    card_paginator = Paginator(all_cards, 25)

    page = req.GET.get('page') or 1
    page_obj = card_paginator.get_page(page)

    return render(req, 'cards/list.html', {'page_obj': page_obj})


def card(req, card_uuid):
    return HttpResponse("You're looking at card {}".format(card_uuid))


def import_library(req):
    if req.user.is_authenticated:
        all_cards = Card.objects.all().filter().annotate(set_name=F('card_set__name')).order_by('name').annotate(
            card_img=Subquery(
                CardFace.objects.filter(
                    card__id=OuterRef('id')
                ).distinct('card__id').values('small_img_uri')

            )
        )
        card_paginator = Paginator(all_cards, 25)

        page = req.GET.get('page') or 1
        page_obj = card_paginator.get_page(page)

        return render(req, 'import_page.html')
    else:
        return redirect('{}?next={}'.format(settings.LOGIN_URL, req.path))


def library(req):
    if req.user.is_authenticated:
        card_ids_owned = CardOwnership.objects.all().filter(user=req.user).values_list('card__id', flat=True)

        all_cards_owned = Card.objects.all().filter(
            id__in=card_ids_owned
        ).annotate(set_name=F('card_set__name')).order_by('name').annotate(
            card_img=Subquery(
                CardFace.objects.filter(
                    card__id=OuterRef('id')
                ).distinct('card__id').values('small_img_uri')

            )
        )

        card_paginator = Paginator(all_cards_owned, 25)

        page = req.GET.get('page') or 1
        page_obj = card_paginator.get_page(page)

        return render(req, 'cards/list.html', {'page_obj': page_obj})
    else:
        return redirect('{}?next={}'.format(settings.LOGIN_URL, req.path))


