from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc

from flask import Flask, render_template, request, redirect, url_for,jsonify
from connection import connection
from passlib.hash import sha256_crypt
from pymysql import escape_string as thwart

from packages.preprocess import preprocesses
from packages.classifier import training
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_FOLDER = './uploads/train/'
PRE_FOLDER = './uploads/pre/'
CLASSIFIER = './class/classifier.pkl'
MODEL_DIR = './models'
npy = ''


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login/', methods=['Get', 'Post'])
def login():
    try:

        c, conn = connection()
        if request.method == "POST":
            username = request.form['username']
            password = request.form['password']
            data = c.execute("SELECT * FROM userdetails WHERE username=(%s)", (thwart(username)))
            data = c.fetchone()[3]
            print(data)
            if sha256_crypt.verify(password, data):
                return redirect(url_for("criminalinfo"))
            else:
                print("Inavalid credentials. try again")

        gc.collect()
        return render_template("login.html")

    except Exception as e:

        print(e)

    return render_template("login.html")


@app.route('/regi/', methods=["GET", "POST"])
def register():
    try:
        c, conn = connection()
        if request.method == "POST":
            username = request.form['username']
            password = sha256_crypt.encrypt(str(request.form['password']))
            fname = request.form['fname']
            mail = request.form['mail']

            data = c.execute("SELECT * FROM  userdetails WHERE username=(%s)", (thwart(username)))

            if int(data) > 0:
                print(data)
            else:
                print("gu")
                c.execute("INSERT INTO  userdetails (fname,username,password,email,status) values(%s,%s,%s,%s,%s)",
                          (thwart(fname), thwart(username), thwart(password), thwart(mail), thwart("Active")))
                print("gu")

                conn.commit()
                c.close()
                conn.close()


    except Exception as e:
        return (str(e))

    return render_template("regi.html")


@app.route('/criminalinfo/', methods=["GET", "POST"])
def criminalinfo():
    try:
        c, conn = connection()
        if request.method == "POST":
            name = request.form['name']
            nic1 = request.form['nic']
            age = request.form['age']
            add01 = request.form['add01']
            add02 = request.form['add02']
            add03 = request.form['add03']
            eye = request.form['eye']
            hair = request.form['hair']

            data = c.execute("SELECT * FROM  criminalinfo WHERE nic=(%s)", (thwart(nic1)))

            if int(data) > 0:
                print(data)
            else:

                c.execute(
                    "INSERT INTO  criminalinfo (nic,name,age,addressline01,addressline02,addressline03,eyecolor,haircolor,gender) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (thwart(nic1), thwart(name), thwart(age), thwart(add01), thwart(add02), thwart(add03), thwart(eye),
                     thwart(hair), thwart("nnn")))
                conn.commit()
                c.close()
                conn.close()
                target = os.path.join(APP_ROOT, 'uploads/train/' + nic1)
                print(target)
                if not os.path.isdir(target):
                    os.mkdir(target)

                count = 0
                for file in request.files.getlist('img'):
                    print(file)
                    count = count + 1
                    filename = file.filename
                    print(filename)
                    destination = "/".join([target, filename])
                    print(destination)
                    file.save(destination)

                # ttt
                obj = preprocesses(TRAIN_FOLDER, PRE_FOLDER)
                nrof_images_total, nrof_successfully_aligned = obj.collect_data()

                print('Total number of images: %d' % nrof_images_total)
                print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

                print("Training Start")
                obj = training(PRE_FOLDER, MODEL_DIR, CLASSIFIER)
                get_file = obj.main_train()
                print('Saved classifier model to file "%s"' % get_file)


               # flash('User registeration succeeded please log in', 's_msg')
                return jsonify(success=["User Registration Success"], value=True)


    except Exception as e:
        return (str(e))
        print(e)
    return render_template("criminalinfo.html")


@app.route('/view/', methods=["GET", "POST"])
def view():
    try:
        c, conn = connection()

        nic = "1"
        data = c.execute("SELECT * FROM  criminalinfo WHERE nic=(%s) ", (thwart(nic)))
        raw = c.fetchone()
        print(raw)
        nic = raw[1]
        name = raw[2]
        age = raw[3]
        address = raw[4]
        address02 = raw[5]
        address03 = raw[6]
        haircolor = raw[7]
        eyecolor = raw[8]

        print(nic)
        return render_template("view.html", nic=nic, name=name, address=address, address02=address02,
                               address03=address03, age=age, haircolor=haircolor, eyecolor=eyecolor)


    except Exception as e:
        return (str(e))


@app.route('/search/', methods=["GET", "POST"])
def search():
    try:
        c, conn = connection()
        if request.method == "POST":
            comit = request.form['commitdate']

            arrest = request.form['arrestdate']
            details = request.form['details']
            nic = request.form['nic']

            print("gu")
            c.execute("INSERT INTO  commitedcrime (nic,commitdate,arresteddate,	details) values(%s,%s,%s,%s)",
                      (thwart(nic), thwart(comit), thwart(arrest), thwart(details)))
            print("gu")

            conn.commit()
            c.close()
            conn.close()


    except Exception as e:
        return (str(e))

    return render_template("regi.html")


if __name__ == "__main__":
    app.run()
