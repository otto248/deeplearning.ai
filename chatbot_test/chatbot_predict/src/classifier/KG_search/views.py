from django.shortcuts import render

# Create your views here.

from . import search
raw_info = search.read_database()

def search(request):
    text = request.get('text')
    matchEntity = search.fuzzymatch(text,raw_info[2])