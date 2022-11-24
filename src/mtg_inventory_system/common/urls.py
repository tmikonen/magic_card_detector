from django.urls import path

from . import views

urlpatterns = [
    path('', views.cards, name='list cards'),
    path('<str:card_uuid>/', views.card, name='card details')
]