import os

# Gunicorn configuration file automatically picked up by Gunicorn
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 1
threads = 2
timeout = 180
worker_class = "sync"
loglevel = "info"
keepalive = 5
