from django.urls import path

from . import views

urlpatterns = [
    path('', views.library, name='User Library'),
    path('all/', views.all_cards, name='List Cards'),
    path('view/<str:card_uuid>/', views.card, name='card_details'),
    path('import_library/', views.import_library, name='Library Import'),
    path('add_to_library/<str:card_uuid>/', views.add_to_library_form, name='add_to_library_form'),
]