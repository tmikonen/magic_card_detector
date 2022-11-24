from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('cards/', views.cards, name='list cards'),
    path('card/<str:card_uuid>/', views.card, name='card details')
]