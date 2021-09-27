from app import app
import os
import urllib.request
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename

IMG_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
VID_EXTENSIONS = set(['mp4'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VID_EXTENSIONS



@app.route("/")
def index():
    return "Hello World"



if __name__ == "__main__":
    app.run()