from django.urls import path

from . import views

urlpatterns = [
    path('', views.LibraryCardsListView.as_view(), name='User Library'),
    path('all/', views.CardsListView.as_view(), name='list_cards'),
    path('view/<str:card_uuid>/', views.card, name='card_details'),
    path('import_library/', views.import_library, name='Library Import'),
    path('add_to_library/<str:card_uuid>/', views.add_to_library_form, name='add_to_library_form'),
    path('clear_library/', views.clear_library, name='clear_library')
]