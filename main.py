from flask import Flask, request, make_response, render_template, redirect, url_for, send_file
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/landing_page",methods=["GET","POST"])
def landing_page():
    return render_template("upload.html", image_path = 'landing_page_pic.jpg')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)