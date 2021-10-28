from flask import Flask

app = Flask(__name__, template_folder='application/templates')

from application import run