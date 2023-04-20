import os
from stability_sdk import client
from programs.prompt import randomPrompt
from flask import Flask, render_template, request
from programs.generateandSaveImage import generate_and_save_image
from programs.classifySound import extract_feature_and_print_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/result', methods= ['POST', 'GET'])
def result():
    if request.method == 'POST':
        #Saving the audio into UPLOAD_FOLDER
        audio = request.files['rawAudio']
        oripath = 'static/audio/audio.wav'
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

        prompt = randomPrompt(pred)

        if generate_and_save_image(stability_api, prompt, path = 'static/images//'):
            return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)