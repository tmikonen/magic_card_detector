from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout


def login_view(request):


def login_req(request):
    username = request.POST['username']
    password = request.POST['password']
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return redirect('/cards')
    else:
        return redirect('%s?next=%s' % (settings.LOGIN_URL, request.path))


def logout_view(request):
    logout(request)
    return redirect(settings.LOGIN_UR)

