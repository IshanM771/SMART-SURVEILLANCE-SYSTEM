from flask import Flask,render_template,request

from face_datasets import collect_training_data
from face_recognition_from_video import face_recognition_from_video
# from flask_mail import Mail

app=Flask(__name__)
# app.config.update(
#     MAIL_SERVER="smtp.gmail.com",
#     MAIL_PORT="465",
#     MAIL_USE_SSL=True,
#     MAIL_USERNAME="harshadag954@gmail.com",
#     MAIL_PASSWORD="Finalyearproject@2024"
# )
# mail=Mail(app)


@app.route("/")
def home():
    # if(request.method=="POST"):
    #     name=request.form.get('name')
    #     email=request.form.get('email')
    #     subject=request.form.get('subject')
    #     message=request.form.get('message')
    #     mail.send_message('New message from ' + name,
    #                         sender = email,
    #                         recipients = "harshadag954@gmail.com",
    #                         body = subject+"\n"+message,
    #                         )

    return render_template("index.html")


@app.route("/service1",methods=['GET','POST'])
def service1():
    if(request.method=="POST"):
        name=request.form.get('name')
        print(name)
        collect_training_data(name)
    return render_template("service1.html")

@app.route("/service2",methods=['GET','POST'])
def service2():
     if(request.method=="POST"):
        print("service2")
        file=request.form.get('filename')
        face_recognition_from_video(file)
     return render_template("service2.html")

app.run(debug=True)



