from django.urls import path

from . import views

urlpatterns = [
    path('', views.library, name='User Library'),
    path('all/', views.cards, name='list cards'),
    path('<str:card_uuid>/', views.card, name='card details'),
    path('import_library/', views.import_library, name='Library Import'),
]