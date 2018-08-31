from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc

from flask import Flask, render_template, request, redirect, url_for, jsonify
from connection import connection
from passlib.hash import sha256_crypt, md5_crypt
from pymysql import escape_string as thwart

from packages.preprocess import preprocesses
from packages.classifier import training
import os

import pickle
import time
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from packages import facenet, detect_face

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_FOLDER = './uploads/train/'
PRE_FOLDER = './uploads/pre/'
CLASSIFIER = './class/classifier.pkl'
MODEL_DIR = './model'
npy = ''
TEST_FOLDER = './uploads/test/'


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


def recognize(filename="img.jpg"):
    image_path = TEST_FOLDER + filename
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            frame_interval = 3
            image_size = 182
            input_image_size = 160

            HumanNames = os.listdir(TRAIN_FOLDER)
            HumanNames.sort()

            print('Loading feature extraction model')
            facenet.load_model(MODEL_DIR)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(CLASSIFIER)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            c = 0

            print('Start Recognition!')
            frame = cv2.imread(image_path, 0)

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            timeF = frame_interval

            if (c % timeF == 0):

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Face Detected: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        # print("emb_array",emb_array)
                        predictions = model.predict_proba(emb_array)
                        print("Predictions ", predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print("Best Predictions ", best_class_probabilities)

                        if best_class_probabilities[0] > 0.6:
                            print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            for H_i in HumanNames:
                                # print(H_i)
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print("Face Recognized: ", result_names)
                                    return str(result_names)
                        else:
                            print('Not Recognized')
                            return False
                else:
                    print('Unable to align')
                    return False

    return False


def random_name():
    name = md5_crypt.encrypt(str(time.time())).split("$")[2]
    return name


@app.route('/identify/', methods=['POST'])
def authenticateUser():

    target = os.path.join(APP_ROOT, 'uploads/test/')
    if not os.path.isdir(target):
        os.mkdir(target)

    filename = random_name() + ".png"
    destination = "/".join([target, filename])

    for file in request.files.getlist('img'):
        print(filename)
        file.save(destination)

    result = recognize(filename)
    os.remove(destination)

    if result is not False:
        return jsonify(value=result)
    else:
        return jsonify(value="0758965123")





@app.route('/find/<string:id>/')
def find_user_details(id):

    c, conn = connection()
    c.execute("SELECT * FROM  criminalinfo WHERE nic=(%s)", (thwart(str(id))))
    result = c.fetchone()
    print (result)

    c.execute("SELECT * FROM  criminalinfo WHERE nic=(%s)", (thwart(str(id))))
    result1 = c.fetchall()
    print(result1)
    return render_template("searchone.html",data=result,result1=result1)


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


@app.route('/find/', methods=["GET", "POST"])
def find():
    return render_template("view.html")


if __name__ == "__main__":
    app.run()
