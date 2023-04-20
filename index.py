import os
from stability_sdk import client
from flask import Flask, render_template, request

#Generate and Save Image
import io
import warnings
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

#Classify Sound
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import librosa

#Prompt
import random

def generate_and_save_image(api, prompt, path, cfg_scale=8.0, noImage=1):
    width, height = 512, 512
    sampler = generation.SAMPLER_K_DPMPP_2M

    # Generate answer
    answer = api.generate(
        prompt=prompt,
        sampler=sampler,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        samples = noImage
    )

    for resp in answer:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Saftety Filters have been triggered. Modify the prompt and try again."
                )
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(path + "image.png")
                return True
            else:
                return False

def extract_feature_and_print_prediction(file_name):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('le.npy')
    modelFile = tf.keras.models.load_model('model.h5')
    audio_data, sample_rate = librosa.load(file_name) 
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T,axis=0)
    pred_fea = np.array([scaled])
    pred_vector = np.argmax(modelFile.predict(pred_fea),axis=-1)
    pred_class = encoder.inverse_transform(pred_vector)
    return pred_class[0]

air_conditioner = ["Amidst a scorching heatwave, an air conditioner stands tall like a beacon of hope. Its cool breeze offers respite to anyone seeking refuge from the sweltering heat. In this dreamlike painting, capture the air conditioner as an oasis of coolness, surrounded by shimmering heatwaves that distort reality. Let your imagination run wild and infuse the scene with a sense of tranquility, as if the air conditioner is a magical portal to a world of eternal coolness.",
                   "Amidst a scorching heatwave, a lone air conditioner stands tall like an oasis of coolness. Its gentle hum and icy breath offer a respite from the sweltering heat outside. As you gaze at the machine, you cant help but wonder about the secrets of its cooling powers and the relief it brings to those seeking refuge from the unforgiving sun.",
                   "Create a vivid and realistic painting of an air conditioner. Show the intricate details of its components, the sleekness of its design, and the coolness it brings to a room on a hot summer day. Use your artistic abilities to make the viewer feel the refreshing breeze it provides",
                   "I would like you to paint me a picture of an air conditioner. Please use your creativity to depict an air conditioner in a way that is visually appealing and interesting. Consider the form, texture, and color of the air conditioner, as well as its context and surroundings. I look forward to seeing your interpretation of this everyday object."]

children_playing= ["I would love for you to paint me a picture of children playing. Imagine a scene where children are outside, perhaps in a park or a playground, engaged in various activities. You could show them running, jumping, climbing, or playing games. Consider the expressions on their faces, the clothes they are wearing, and the environment around them. Please create a vibrant and lively scene that captures the joy and energy of childhood play. I am excited to see your artistic interpretation!",
            "Can you please paint me a picture of children playing in the snow? Picture a snow-covered landscape with children running, sliding, and building snowmen. Imagine the snowflakes falling gently from the sky, and the laughter and excitement of the children as they play. You could incorporate colorful winter clothing, sleds, and snowballs into the scene, and perhaps show some children sipping hot cocoa to warm up. Please create a joyful and lively scene that captures the magic of winter play. I can't wait to see your artistic interpretation!",
            "I would like you to paint me a picture of children playing at the beach. Picture a scene where the children are building sandcastles, collecting seashells, and splashing in the waves. You could show some children flying kites or playing beach volleyball in the background. Consider the bright colors of beach towels, swimsuits, and umbrellas, and the vastness of the ocean and sky. Please create a lively and playful scene that captures the carefree spirit of childhood at the beach. I am excited to see your artistic interpretation",
            "I would love for you to paint me a picture of children playing in a field of flowers. Imagine a meadow filled with colorful flowers, and children running through the field, picking flowers, and playing games. Consider the bright and vibrant colors of the flowers, the softness of the grass, and the joy and laughter on the faces of the children. You could include elements such as a picnic blanket, a frisbee, or a kite to add to the playful atmosphere. Please create a scene that captures the happiness and wonder of childhood, surrounded by the beauty of nature."]

car_horn =["I would like you to paint me a picture of car horns. Picture a busy street with cars honking their horns. Consider the different tones and rhythms of the horns, and the way they create a symphony of sound. You could show the cars themselves, or perhaps just the horns floating in the air. Consider the colors and textures of the street and the cars, and the way the honking horns add to the energy and chaos of the scene. Please create a dynamic and visually interesting scene that captures the essence of car horns in a city.",
           "Paint a picture of a chaotic city intersection with cars honking their horns in frustration. The scene should convey a sense of rush hour traffic, with multiple cars and trucks jostling for position and impatiently blaring their horns. The colors should be bright and bold, with the honking horns adding a layer of sound and energy to the painting.",
           "Paint a picture of a classic car show, with vintage cars from different eras lined up and their owners proudly showing them off. In the foreground of the painting, there should be a close-up of a classic car's horn, polished and gleaming, with intricate details that capture the craftsmanship of a bygone era. The background should be filled with other cars and people admiring them, but the focus should be on the beauty of the classic car horn. The colors should be rich and vibrant, with the horn shining in the light and the surrounding elements adding to the nostalgic atmosphere.",
           "Paint a picture of a vintage car with a unique and distinctive horn? The car should be set against a beautiful and scenic backdrop, perhaps with rolling hills or a sunset in the distance. The horn should be the centerpiece of the painting, with intricate details and a sense of elegance and charm. The colors should be warm and inviting, and the overall tone of the painting should convey a sense of nostalgia and a bygone era"]

dog_bark = ["Paint a lively scene of dogs barking in a park? The painting should capture the joy and energy of the dogs as they run and play, with their barks filling the air. The colors should be bright and vibrant, with the green grass and blue sky providing a beautiful backdrop for the playful pups. You could also include their owners, perhaps sitting on a bench and enjoying the lively atmosphere. Overall, the painting should convey a sense of happiness and the simple pleasures of spending time with our furry friends.",
            "Paint a picture of a pack of dogs barking excitedly in a natural setting, such as a forest or park. The dogs should be a mix of different breeds and sizes, with their tongues lolling out and ears perked up in excitement. The colors of the painting should be earthy and natural, with the scenery complementing the dogs' energy and liveliness. The dogs' barks should be visible in the painting, perhaps in the form of sound waves or speech bubbles, adding a playful and whimsical touch to the overall composition.",
            "Paint a picture of a pack of playful dogs barking joyously in a grassy field. The scene should be full of energy and movement, with the dogs leaping and bounding through the field as they bark and play. The colors should be vibrant and lively, with the green grass contrasting with the bright and varied fur of the dogs. The painting should capture the joy and excitement of these lovable animals, and convey a sense of their unique personalities and characteristics through their playful barking",
            "Paint a picture of a group of dogs barking excitedly in a park or open field? The dogs should be a mix of breeds and sizes, all with their tongues lolling out and their tails wagging. The background should be lush and green, with perhaps a few trees or flowers in the distance. The dogs should be the main focus of the painting, with their barks conveyed through dynamic brushstrokes and vivid colors. The overall effect should be one of joy and playfulness, capturing the exuberance and energy of our furry friends."]

drilling = ["Paint a picture of the action of drilling, perhaps in an industrial setting? The painting should convey a sense of power and precision, with the drill bit digging into metal or concrete with force. The colors should be cool and metallic, with shades of silver, grey, and black dominating the palette. You could include workers or machinery in the painting to add to the sense of activity and industry, and the background could be a gritty, urban landscape. Overall, the painting should capture the intensity and skill required for the task of drilling.",
            "Paint a picture of a worker drilling into a hard surface, like concrete or metal? The worker should be depicted in motion, with the drill held firmly in their hand and the tool emitting sparks or dust as it bores into the surface. The setting should be a construction site or workshop, with other tools and machinery visible in the background. The colors should be bold and industrial, with shades of gray, silver, and yellow dominating the painting. The overall effect should convey the power and precision of the drilling process, capturing the raw energy and force of the worker and the machine",
            "Paint a picture of a worker operating a large drill in an industrial setting? The focus should be on the action of drilling itself, with the worker depicted in the midst of the process. The painting should convey a sense of power and force, with the drill cutting through a solid surface and throwing up sparks or dust. The background should be gritty and industrial, with perhaps other workers or machinery in the distance. The colors should be bold and intense, conveying the raw energy of the drilling process. Overall, the painting should capture the rugged and intense nature of industrial work.",
            "Paint a picture that captures the action of drilling, with a sense of motion and power? The painting should depict a large drilling machine in operation, perhaps at a construction site or oil rig. The drill should be shown boring into the ground or rock, with debris flying off in all directions. The colors should be intense and bold, with a sense of heat and energy emanating from the drill. The scene should convey a sense of purpose and determination, with the drilling machine symbolizing human ingenuity and the drive to push beyond our limits"]

engine_idling = ["Paint a picture of an idle engine, conveying the stillness and calm of an object at rest? The engine should be the centerpiece of the painting, perhaps shown in close-up detail, with its various parts and components on display. The colors should be muted and tranquil, with a sense of quietude and peacefulness permeating the scene. You might consider depicting the engine in a garage or workshop setting, with tools and equipment in the background, to further convey a sense of stillness and introspection. The overall effect should be one of contemplation and serenity, celebrating the beauty and elegance of even the most mundane objects.",
                 "Paint a picture of an idle engine, conveying a sense of stillness and quiet? The engine could be from a car, a plane, or a boat, and should be shown at rest, with no movement or signs of activity. The colors should be cool and muted, with perhaps a hint of rust or wear and tear, conveying the sense of a machine that has been used and has now come to a peaceful stop. The painting should convey a sense of calm and tranquility, perhaps evoking a moment of quiet reflection or introspection",
                 "Paint a picture of an idle vehicle engine, capturing the intricate details and machinery in a moment of stillness? The engine could be from a car, truck, or motorcycle, and should be shown at rest, with no movement or signs of activity. The colors should be rich and warm, with a sense of depth and texture that conveys the complexity of the engine's parts. The painting should focus on the beauty of the machine itself, perhaps depicting the interplay of light and shadow across its surfaces, and conveying a sense of the machine's potential power even in its stillness.",
                 "Paint a picture of an idle vehicle engine, conveying a Paint a picture of an idle vehicle engine, conveying sense of anticipation and potential energy? The engine could be from a car, a motorcycle, or a truck, and should be shown in close-up, with no other details distracting from its power and complexity. The colors should be rich and deep, with perhaps a hint of shimmer or sheen, conveying the sense of a machine waiting to be put to use. The painting should convey a sense of excitement and anticipation, perhaps evoking the feeling of a driver or rider waiting for the right moment to unleash the full power of the machine."]

gun_shot = ["I want you to paint me a picture of a gun being fired. ",
            "Bullet being fired",
            "Cop shooting a person",
            "Gunshot"]

jackhammer = ["Paint a picture of a jackhammer in action, conveying a sense of power and force? The painting should depict a construction worker using the jackhammer to break through a thick layer of concrete or asphalt. The worker should be shown with muscles tensed and sweat glistening on their forehead, conveying the sense of physical exertion required to operate the tool. The colors should be bold and gritty, with a sense of dust and debris flying off in all directions. The scene should convey a sense of progress and determination, with the jackhammer symbolizing human ingenuity and the drive to overcome obstacles",
              "Paint a picture of a jackhammer in action, conveying a sense of power and force? The jackhammer should be shown in close-up, with debris flying off in all directions as it pounds into the ground. The colors should be bold and dynamic, with a sense of heat and energy emanating from the machine. The scene should convey a sense of industrial might, with the jackhammer symbolizing the power of human innovation and determination. The painting should capture the intensity and relentlessness of the jackhammer's motion, perhaps evoking the feeling of being on a construction site with heavy machinery in action",
              "Paint a picture that captures the power and energy of a jackhammer in action? The painting should depict a construction worker using a jackhammer to break up concrete or pavement, with the machine pounding into the ground and debris flying in all directions. The colors should be bold and intense, with a sense of heat and energy emanating from the jackhammer. The scene should convey a sense of hard work and determination, with the worker symbolizing the strength and ingenuity of human labor",
              "Paint a picture of a jackhammer in action, with a sense of power and motion? The jackhammer should be shown breaking up concrete or asphalt, with debris flying off in all directions. The worker operating the jackhammer should be shown in the background, perhaps in silhouette or partially obscured by the dust and debris. The colors should be bold and intense, with a sense of heat and energy emanating from the jackhammer. The painting should convey a sense of strength and determination, capturing the hard work and perseverance required to break through barriers and overcome obstacles"]

siren = ["A siren made by a police car",
         "A siren made by an ambulance",
         "A siren made by a Fire truck",
         "Police, Ambulance and Fire truck siren"] 

street_music = ["Paint a picture that captures the vibrant energy and joy of street music? The painting should depict a street performer playing an instrument or singing, surrounded by a crowd of people listening and enjoying the music. The colors should be bright and bold, conveying the sense of excitement and movement on the street. The scene should convey a sense of community and togetherness, with people from all walks of life coming together to enjoy the music. The painting should capture the feeling of a moment of shared happiness and celebration, and the power of music to bring people together.",
                "Paint a picture that captures the joy and energy of street music? The painting should depict a scene with musicians playing their instruments on a busy city street corner, with people gathered around to listen and enjoy the music. The colors should be bright and lively, with a sense of movement and rhythm conveyed through the brushstrokes. The scene should capture the vibrancy and diversity of a bustling city, with different people and cultures coming together to share in the joy of music. The painting should convey a sense of community and connection, and the power of music to bring people together.",
                "Paint a picture that captures the vibrant energy of street music? The painting should depict a street musician or group of musicians playing their instruments in a public space, perhaps in a bustling city square or on a charming cobblestone street. The colors should be warm and inviting, with a sense of joy and celebration emanating from the music. The scene should capture the diverse crowd of people stopping to listen and dance to the music, with perhaps a few local vendors or performers adding to the festive atmosphere. The painting should convey a sense of community and togetherness, with the power of music bringing people from all walks of life together in a shared experience of joy and beauty.",
                "Paint a picture of a street musician playing music, with a sense of joy and celebration? The musician could be playing any type of instrument, perhaps a guitar, a saxophone, or a violin. The painting should depict a lively scene, with people stopping to listen and perhaps dancing along to the music. The colors should be warm and inviting, with perhaps a hint of dusk or sunset in the background, conveying the sense of a vibrant street scene. The painting should capture the energy and spontaneity of street music, and the sense of community that can be built around shared experiences of music and culture."]

def randomPrompt(var):
    if var == 'air_conditioner':
        return random.choice(air_conditioner)
    elif var == 'children_playing':
        return random.choice(children_playing)
    elif var == 'car_horn':
        return random.choice(car_horn)
    elif var == 'dog_bark':
        return random.choice(dog_bark)
    elif var == 'drilling':
        return random.choice(drilling)
    elif var == 'engine_idling':
        return random.choice(engine_idling)
    elif var == 'gun_shot':
        return random.choice(gun_shot)
    elif var == 'jackhammer':
        return random.choice(jackhammer)
    elif var == 'siren':
        return random.choice(siren)
    elif var == 'street_music':
        return random.choice(street_music)

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


# if __name__ == '__main__':
#     app.run(debug=True)