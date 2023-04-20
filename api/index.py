import os
from stability_sdk import client
from flask import Flask, render_template, request
from programs.generateandSaveImage import generate_and_save_image
from programs.classifySound import extract_feature_and_print_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/result', methods= ['POST', 'GET'])
def result():
    if request.method == 'POST':
        #Saving the audio into UPLOAD_FOLDER
        audio = request.files['rawAudio']
        oripath = app.config['UPLOAD_FOLDER'] + 'audio.wav'
        audio.save(oripath)

        #Making prediction off the audio
        pred = extract_feature_and_print_prediction(oripath)

        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        os.environ['STABILITY_KEY'] = 'sk-vLF73x0zUJ5JE8IOibzk3GTIkNgUP8dNKgW1NLLvDo8Tznju'

        # Set up our connection to the API.
        stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'], 
            verbose=True, 
            engine="stable-diffusion-xl-beta-v2-2-2", 
        )

        generate_and_save_image(stability_api, pred, path = app.config['UPLOAD_FOLDER'] + '/images/')

# if __name__ == '__main__':
#     app.run(debug=True)