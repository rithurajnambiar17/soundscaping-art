import io
import warnings
from PIL import Image
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


def generate_and_save_image(api, prompt, cfg_scale, noImage):
    width, height = 512, 512
    sampler = generation.SAMPLER_K_DMPP_2M

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
                img.save(str(prompt) + ".png")
                return True
            else:
                return False