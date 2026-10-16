# -*- coding: utf-8 -*-
"""
Render.com / Gunicorn icin WSGI giris noktasi.
"""
from standup_app import app

if __name__ == '__main__':
    app.run()
