from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def index_model_versus(request):
    context = {
        'svm_normal': {
            'precision': 95.61,
            'recall': 95.67,
            'f1': 95.61,
        },
        'svm_boost': {
            'precision': 97.82,
            'recall': 97.79,
            'f1': 97.8,
        },
        'nb_normal': {
            'precision': 76.49,
            'recall': 76.11,
            'f1': 76.27,
        },
        'nb_boost': {
            'precision': 82.6,
            'recall': 82.56,
            'f1': 82.46,
        },
    }
    return render(request, template_name="models_versus/index.html", context=context)
