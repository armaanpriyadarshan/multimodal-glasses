import requests
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from flask import Flask, request
from flaskngrok import runwithngrok
from io import BytesIO
import string

CKPTDIR = "OFA-medium"

tokenizer = OFATokenizer.frompretrained(CKPTDIR)
model = OFAModel.from_pretrained(CKPT_DIR, use_cache=False)

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

app = Flask(__name)
run_with_ngrok(app)


def answer_question(image, question="what does the image describe?"):
    inputs = tokenizer([question], return_tensors="pt").input_ids
    letters = string.ascii_lowercase
    patch_image = patch_resize_transform(image).unsqueeze(0)

    gen = model.generate(inputs, patch_images=patch_image, num_beams=5, no_repeat_ngram_size=3)

    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0][1:]


@app.route('/multimodal-vision', methods=['GET'])
def multimodal_vision():
    r = requests.get("http://192.168.86.38/capture")
    image = Image.open(BytesIO(r.content))

    question_param = request.args.get("question")
    if question_param is not None:
        return answer_question(image, question_param)
    else:
        return answer_question(image)


if __name == "__main":
    app.run()
