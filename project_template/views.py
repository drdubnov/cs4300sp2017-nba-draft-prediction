from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Create your views here.
def index(request):
    output_list = ''
    output=''
    position = ''
    search = ''
    version = ''
    if request.GET.get('search'):
        search = request.GET.get('search')
        short_pos = request.GET.get('positions')
        version = request.GET.get('version')
        if short_pos == "any":
            position = "Output for Players of Any Position"
        elif short_pos == "pg":
            position = "Output for Point Guards"
        elif short_pos == "sg":
            rposition = "Output for Shooting Guards"
        elif short_pos == "sf":
            position = "Output for Small Forwards"
        elif short_pos == "pf":
            position = "Output for Power Forwards"
        elif short_pos == "c":
            position = "Output for Centers"
        output_list = find_similar(search, short_pos, version)
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
    return render_to_response('project_template/index.html', 
                          {'output': output,
                           'pos': position,
                           'search': '"' + search + '"',
                           'magic_url': request.get_full_path(),
                           })

    