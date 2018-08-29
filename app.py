import gc

from flask import Flask, render_template, request, redirect, url_for
from connection import connection
from passlib.hash import sha256_crypt
from pymysql import escape_string as thwart

app = Flask(__name__)


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
                    (thwart(name), thwart(nic1), thwart(age), thwart(add01), thwart(add02), thwart(add03), thwart(eye),
                     thwart(hair), thwart("nnn")))


                conn.commit()
                c.close()
                conn.close()


    except Exception as e:
        return (str(e))
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
        return render_template("view.html", nic=nic, name=name, address=address, address02=address02,address03=address03, age=age, haircolor=haircolor, eyecolor=eyecolor)


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
            nic= request.form['nic']

            print("gu")
            c.execute("INSERT INTO  commitedcrime (nic,commitdate,arresteddate,	details) values(%s,%s,%s,%s)",
                          (thwart(nic),thwart(comit), thwart(arrest), thwart(details)))
            print("gu")

            conn.commit()
            c.close()
            conn.close()


    except Exception as e:
         return (str(e))

    return render_template("regi.html")

if __name__ == "__main__":
    app.run()
