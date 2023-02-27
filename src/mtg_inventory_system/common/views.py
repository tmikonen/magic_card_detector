import json
import logging
from collections import OrderedDict

from django.conf import settings
from django.core.paginator import Paginator

from django.db.models import F, Q, Subquery, OuterRef
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect

from django.views import View
from django.views.generic import ListView, DetailView

from .forms import *
from .models import Card, CardFace, CardOwnership, CardPrice

logger = logging.getLogger(__name__)


class AuthViewMixin(View):
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            response = redirect('{}?next={}'.format(settings.LOGIN_URL, request.path))
        else:
            response = super(AuthViewMixin, self).dispatch(request, *args, **kwargs)

        return response


def index(req):
    return HttpResponse("Hello, world. You're at the common index.")


class CardsListView(ListView):
    model = Card
    template_name = 'cards/list.html'
    paginate_by = 25

    def get_queryset(self):
        result = super(CardsListView, self).get_queryset().order_by('name', 'card_set__name').distinct('name') \
            .annotate(
            card_img=Subquery(
                CardFace.objects.filter(
                    card__id=OuterRef('id')
                ).distinct('card__id').values('small_img_uri')
            )
        )
        query = self.request.GET.get('search')
        if query:
            post_result = result.filter(
                Q(name__icontains=query) |
                Q(cardface__type_line__icontains=query) |
                Q(cardface__oracle_text__icontains=query))
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
        result['price_data'] = usd_card_price_chart_data(result['card'].name)

        # General Library Details
        if self.request.user.is_authenticated:
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


def usd_card_price_chart_data(card_name):
    def _price_obj(set_name:str, prices:list[float]):
        return {
            'label': set_name,
            'data': prices
        }

    def _printing_to_date_obj(dates, printings):
        mapping_obj = OrderedDict()
        for printing in printings:
            mapping_obj[printing] = {}
            for date in dates:
                mapping_obj[printing][date] = 0

        return mapping_obj


    price_data = CardPrice.objects.filter(card__name=card_name).values(
        'date',
        'price_usd',
        'card_id',
        set_name=F('card__card_set__name'),
        collector_number=F('card__collector_number'),
    ).order_by('date')

    all_dates_set = set(price_data.values_list('date', flat=True))
    all_printings_set = {"{} #{}".format(obj['set_name'], obj['collector_number']) for obj in price_data}

    printings_to_dates = _printing_to_date_obj(all_dates_set, all_printings_set)

    # Note this assumes we have one price history for each day and that all printings have the same
    for data in price_data:
        printings_to_dates[f'{data["set_name"]} #{data["collector_number"]}'][data["date"]] = float(data.get('price_usd') or 0)

    labels = [str(d) for d in all_dates_set]
    datasets = [_price_obj(card_set, list(prices.values())) for card_set, prices in printings_to_dates.items()]
    # prices = [float(price or 0) for price in price_data.values_list('price_usd', flat=True)]
    data = {
        'labels': labels,
        'datasets': datasets
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
        return result


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
