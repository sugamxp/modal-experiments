import io
from pathlib import Path

from modal import (
    Image,
    Mount,
    Stub,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)

sdxl_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
      "invisible_watermark==0.2.0",
      "Pillow~=10.1.0",
      "diffusers~=0.24.0",
      "transformers~=4.35.2",  # This is needed for `import torch`
      "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
      "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
    )
)

stub = Stub("stable-diffusion-xl")

with sdxl_image.imports():
    import torch
    from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
    from fastapi import Response
    from huggingface_hub import snapshot_download
    from diffusers.utils import load_image
    from PIL import Image

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=sdxl_image)
class Model:
    @build()
    def build(self):
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )
        
        #load img2img model
        self.img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()
        
        
    def _inference_img2img(self, image_bytes, prompt):
        print("Image type", type(Image.open(io.BytesIO(image_bytes))))
        pil_image = Image.open(io.BytesIO(image_bytes))
        init_image = load_image(pil_image)
        image = self.img2img(
            prompt,
            image=init_image
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")

        return byte_stream
      
    @method()
    def inference_img2img(self, image_bytes, prompt):
        return self._inference_img2img(
            image_bytes, prompt
        ).getvalue()
        

    @web_endpoint()
    def web_inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return Response(
            content=self._inference(
                prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
            ).getvalue(),
            media_type="image/jpeg",
        )
        
    @web_endpoint(label="img2img")
    def web_inference_img2img(self, image_bytes, prompt):
        return Response(
            content=self._inference_img2img(
                image_bytes, prompt
            ).getvalue(),
            media_type="image/jpeg",
        )

DEFAULT_IMAGE_PATH = Path(__file__).parent / "profile.png"


@stub.local_entrypoint()
def main(prompt: str, image_path: str = None):
  
  if image_path is None:
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
        
  else:
    with open(DEFAULT_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
    image_bytes = Model().inference_img2img.remote(image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)