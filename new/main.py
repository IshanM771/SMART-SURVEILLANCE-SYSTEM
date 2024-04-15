from flask import Flask, Response,render_template,request,redirect, flash
from werkzeug.utils import secure_filename
# from flask_mail import Mail
import os
from face_datasets import collect_training_data
from loggen import performFaceRecognitionfromVideo
from training import train_model
from file_read_backwards import FileReadBackwards

UPLOAD_FOLDER = 'videos/'
ALLOWED_EXTENSIONS = {'mp4'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "xyz"

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
        if request.form['action'] == 'collect':
            name=request.form.get('name')
            collect_training_data(name)
            flash('Images captured successfully,Now you can train model..')
                        
        if request.form['action'] == 'train':            
            try:
                flash('Training Model..wait')
                train_model()
                flash('Model training done..')
            except Exception as e:
                print("Exception:",str(e))
                flash("Training Failed due to some internal exception")

            return render_template("service1.html")
            
    return render_template("service1.html")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/service2",methods=['GET','POST'])
def service2():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('File is not attached..Attach file first')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("success")

            performFaceRecognitionfromVideo(file.filename)
            with FileReadBackwards("identification_log.txt") as f:
                # getting lines by lines starting from the last line up
                b_lines = [ row for row in f ]

            return render_template('service2.html', b_lines=b_lines)
        else:
            flash('Only video file is allowed')
            return redirect(request.url)
        
    return render_template("service2.html")

app.run(debug=True)



