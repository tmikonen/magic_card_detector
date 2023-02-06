import json
import logging
import time

from django.contrib.auth.models import User
from django.conf import settings
from django.contrib.auth.models import User
from django.core.paginator import Paginator

from django.db.models import F, Q, Value, URLField, Subquery, OuterRef, Count
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import loader
from django.urls import reverse
from urllib.parse import urlencode

from django.views import View
from django.views.generic import CreateView, ListView, DetailView

from .forms import *
from .models import Card, CardFace, CardOwnership, CardPrice

logger = logging.getLogger(__name__)


class AuthViewMixin(View):
    def dispatch(self, request, *args, **kwargs):
        response = super(AuthViewMixin, self).dispatch(request, *args, **kwargs)

        if not request.user.is_authenticated:
            response = redirect('{}?next={}'.format(settings.LOGIN_URL, request.path))

        return response


def index(req):
    return HttpResponse("Hello, world. You're at the common index.")


class CardsListView(ListView):
    model = Card
    template_name = 'cards/list.html'
    paginate_by = 25

    def get_queryset(self):
        result = super(CardsListView, self).get_queryset().annotate(set_name=F('card_set__name')).order_by('name') \
            .annotate(
            card_img=Subquery(
                CardFace.objects.filter(
                    card__id=OuterRef('id')
                ).distinct('card__id').values('small_img_uri')

            )
        )
        query = self.request.GET.get('search')
        if query:
            post_result = Card.objects.filter(
                Q(name__icontains=query) |
                Q(cardface__type_line__icontains=query) |
                Q(cardface__oracle_text__icontains=query)
            ) \
                .annotate(set_name=F('card_set__name')).order_by('name') \
                .annotate(
                card_img=Subquery(
                    CardFace.objects.filter(
                        card__id=OuterRef('id')
                    ).distinct('card__id').values('small_img_uri')

                )
            )
            result = post_result
        return result


class CardDetailView(DetailView):
    model = Card
    template_name = "cards/view_card.html"

    def get_context_data(self, **kwargs):
        result = super(CardDetailView, self).get_context_data(**kwargs)
        # General Card Details
        result['image_uris'] = CardFace.objects.filter(card_id=result['card'].pk) \
            .values_list('normal_img_uri', flat=True)
        result['set_name'] = result['card'].card_set.name
        result['card_id'] = result['card'].id

        # Card prices
        price_data = CardPrice.objects.filter(card_id=result['card']).values('date', 'price_usd').order_by('date')
        price_found = False
        for price in list(price_data):
            if price['price_usd']:
                result['recent_cost_date'] = str(price['date'])
                result['recent_cost'] = float(price['price_usd'])
                result['currency'] = '(USD)'
                result['currency_symbol'] = '$'
                price_found = True
        if not price_found:
            result['recent_cost'] = 'N/A'
            result['recent_cost_date'] = price_data.first()['date']
        result['price_data'] = usd_card_price_chart_data(result['card_id'])

        # General Library Details
        ownership_objs = CardOwnership.objects.filter(user=self.request.user, card__id=result['card'].id)
        if ownership_objs:
            lib_details = {
                'count': ownership_objs.count(),
                'avg_purchase': sum(CardOwnership.objects.filter(
                    user=self.request.user,
                    card__id=result['card'].id).values_list('price_purchased', flat=True)) / ownership_objs.count()
            }
            result['library_details'] = lib_details

        return result


def usd_card_price_chart_data(card_uuid):
    price_data = CardPrice.objects.filter(card_id=card_uuid).values('date', 'price_usd').order_by('date')
    labels = [str(d) for d in price_data.values_list('date', flat=True)]
    prices = [float(price or 0) for price in price_data.values_list('price_usd', flat=True)]
    data = {
        'labels': labels,
        'datasets': [
            {
                'label': 'Price USD',
                'data': prices
            }
        ]
    }

    return json.loads(json.dumps(data))


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


class LibraryCardsListView(CardsListView, AuthViewMixin):
    model = Card
    template_name = 'cards/library.html'
    paginate_by = 25

    def get_queryset(self):
        result = super(LibraryCardsListView, self).get_queryset().filter(cardownership__user=self.request.user)
        aggregated_results = result.annotate(id_count=Count('id'))
        return aggregated_results


def add_to_library_form(req, card_uuid):
    if req.user.is_authenticated:
        if req.method == 'POST':
            form = CreateCardOwnershipForm(req.POST)
            form.card = Card.objects.get(id=card_uuid)
            if form.is_valid():
                form.add_to_library(req.user)
                return HttpResponseRedirect('/cards/')
        else:
            form = CreateCardOwnershipForm()
            form.card = Card.objects.get(id=card_uuid)

        return render(req, 'forms/add_to_library.html', {'form': form})
    else:
        return redirect('{}?next={}'.format(settings.LOGIN_URL, req.path))


def clear_library(req):
    if req.user.is_authenticated:
        req.user.cardownership_set.all().delete()
        return redirect('/cards/')
    else:
        return redirect('{}?next={}'.format(settings.LOGIN_URL, req.path))
