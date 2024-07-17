from django.shortcuts import render

def home(request):
    # Add any logic or context data if needed
    return render(request, 'index.html')