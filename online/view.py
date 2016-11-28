from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import bayes
import random

app = Flask(__name__)
Bootstrap(app)

prior, condprob = bayes.runBayes()


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", active=0)


@app.route("/check", methods=['POST', 'GET'])
def checkmsg():
    msg = request.form['msg']
    result = bayes.predict(msg, prior, condprob)
    return str(result)


@app.route("/getmsg")
def randomMsg():
    index = random.randint(0, 20000)
    with open("data/不带标签短信.txt", encoding='utf-8') as file:
        return file.readlines()[index]


@app.route("/project")
def project():
    return render_template("project.html", active=1)


@app.route("/algorithm")
def algorithm():
    return render_template("algorithm.html", active=2)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run("0.0.0.0")
