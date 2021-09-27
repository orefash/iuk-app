import os
import urllib.request
from flask import Flask, render_template, Response, request, jsonify, abort, make_response
from bson.json_util import dumps
from bson.json_util import loads
import pdfkit
from io import BytesIO

from db import db, testDB, userDB
from model import *
from werkzeug.utils import secure_filename
from vprocess import calib_func, handle_move

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "glory2"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#
# @app.route("/testdb")
# def test():
#     umov = Umovements("orefash@gmail.com")
#
#     jsonstr = umov.__dict__
#     print(jsonstr)
#     userDB.insert_one(jsonstr)
#     cursor = list(userDB.find())
#     print(loads(dumps(cursor)))
#     return "Connected to the data base!"
#
#
# @app.route("/test_find")
# def test_find():
#
#     email = "a@mail.com"
#
#     result = userDB.find_one({"email": email})
#     print(result)
#     if result is None:
#         umov = Umovements(email)
#         jsonstr = umov.__dict__
#         userDB.insert_one(jsonstr)
#         cursor = list(userDB.find())
#         print(loads(dumps(cursor)))
#
#     return "Connected to the data base!"
#
#
# @app.route("/test_update")
# def test_update():
#
#     email = "orefash1@gmail.com"
#     move = "sit2stand"
#     from model import Sit2stand
#     from db import add_movement_record
#     s2s = Sit2stand(200).__dict__
#
#     status = add_movement_record(move, s2s, email)
#
#     print("Status: ", status)
#
#     return "Connected to the data base!"


@app.route('/main-upload', methods=['POST'])
def main_upload():
    # global status
    status = 0
    email = request.form["email"]
    movement = request.form["move"]
    cal = request.form["ppm"]

    print("Email + movem; ppm: ", email+" "+movement+" "+str(cal))

    uploaded_file = request.files['video']
    filename = secure_filename(email+movement+".mp4")

    print("file: ", filename)

    # try:
    if filename != '':
        # file_ext = os.path.splitext(filename)[1]
        # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        #     abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        vpath = "./uploads/" + filename
        # res = handle_move(movement, 500, vpath, email)
        res = handle_move(movement, 500, "./demo/nc_sit.mp4", email)

        print("In exex: ", res)

    return jsonify({"status": 0})


@app.route('/calibrate-upload', methods=['POST'])
def calibrate_upload():
    email = request.form["email"]
    movement = request.form["move"]

    print("Email + movem: ", email+" "+movement)

    uploaded_file = request.files['video']
    filename = secure_filename(email+"calib.mp4")

    print("file: ", filename)
    global ppm
    ppm = 0
    # try:
    if filename != '':
        # file_ext = os.path.splitext(filename)[1]
        # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        #     abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        vpath = "./uploads/" + filename
        ppm = calib_func(vpath)

    return jsonify({"status": 0, "ppm": ppm})

#
# @app.route("/add_one")
# def add_one():
#     db.iuk.insert_one({'title': "todo title", 'body': "todo body"})
#     return jsonify(message="success")


@app.route("/report", methods=['POST'])
def report():

    email = request.form["email"]
    result = userDB.find_one({"email": email})
    print(result)

    global s2s
    s2s = None
    if len(result['sit2stand']) > 0:
        s2s = result['sit2stand'][-1]
        print("s2s: ", s2s)

    global hip_abduct
    hip_abduct = None
    if len(result['hip_abduction']) > 0:
        hip_abduct = result['hip_abduction'][-1]
        print("hip_abd: ", hip_abduct)

    global cerv_rot
    cerv_rot = None
    if len(result['cerv_rot']) > 0:
        cerv_rot = result['cerv_rot'][-1]
        print("cerv_rot: ", cerv_rot)

    global lumbar_flex
    lumbar_flex = None
    if len(result['lumbar_flex']) > 0:
        lumbar_flex = result['lumbar_flex'][-1]
        print("lumbar_flex: ", lumbar_flex)

    global shoulder_flex
    shoulder_flex = None
    if len(result['shoulder_flex']) > 0:
        shoulder_flex = result['shoulder_flex'][-1]
        print("shoulder_flex: ", shoulder_flex)

    global hip_rot
    hip_rot = None
    if len(result['hip_int_rot']) > 0:
        hip_rot = result['hip_int_rot'][-1]
        print("hip_int_rot: ", hip_rot)

    global side_flex
    side_flex = None
    if len(result['side_flex']) > 0:
        side_flex = result['side_flex'][-1]
        print("side_flex: ", side_flex)

    global tragus
    tragus = None
    if len(result['tragus']) > 0:
        tragus = result['tragus'][-1]
        print("tragus: ", tragus)

    html = render_template("report.html", email=email, tragus=tragus, side_flex=side_flex, hip_rot=hip_rot,
                           hip_abduct=hip_abduct, shoulder_flex=shoulder_flex, lumbar_flex=lumbar_flex,
                           cerv_rot=cerv_rot, s2s=s2s)
    config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    # pdfkit.from_string(html, 'MyPDF.pdf', configuration=config)
    pdfd = pdfkit.from_string(html, False, configuration=config)

    response = make_response(pdfd)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "inline; filename=MyReport.pdf"
    return response


@app.route("/")
def demo():
    return render_template("indexCV.html")


@app.route("/main-record", methods=['POST'])
def main_record():
    email = request.form["email"]
    movement = request.form["movement"]
    calibrat = request.form["ppm"]

    print("Email + movem+ calibrate: ", email+" "+movement+" "+str(calibrat))

    return render_template("main_recording.html", email=email, movement=movement, ppm=calibrat)


@app.route("/calibrate", methods=['POST'])
def calibrate():
    email = request.form["email"]
    movement = request.form["movement"]

    result = userDB.find_one({"email": email})
    print("Check: ", result)
    if result is None:
        umov = Umovements(email)
        jsonstr = umov.__dict__
        userDB.insert_one(jsonstr)
        cursor = list(userDB.find())
        print(loads(dumps(cursor)))

    print("Email + movem: ", email+" "+movement)

    return render_template("calibration_recording.html", email=email, movement=movement)

#
# @app.route("/")
# def index():
#     return "Hello World"


if __name__ == "__main__":
    # app.run(ssl_context='adhoc')
    app.run()
