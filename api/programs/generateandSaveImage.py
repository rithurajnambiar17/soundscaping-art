import io
import warnings
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

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