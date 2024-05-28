from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


def login_view(request):
    if request.user.is_authenticated:
        # Si el usuario ya está autenticado, redirigir a la página de inicio
        return redirect('home')

    if request.method == 'POST':
        print(request.POST)
        username = request.POST['user']
        password = request.POST['password']

        # Autenticar al usuario
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')  # Redirigir a la página principal u otra vista
        else:
            messages.error(request, 'Nombre de usuario o contraseña incorrectos')
            return redirect('login')

    return render(request, template_name="login/login.html")


def logout_view(request):
    logout(request)
    return redirect('login')
