from PIL import Image
from torchvision import transforms
import torch
from transformers import OFATokenizer, OFAModel
from OFA.transformers.src.transformers.models.ofa.generate.sequence_generator import SequenceGenerator

ckpt_dir = "OFA-medium"
path_to_image = "C:\Users\armaa\Downloads\negus.jpg"

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

txt = "what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
img = Image.open(path_to_image)
patch_img = patch_resize_transform(img).unsqueeze(0)

model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)

print(tokenizer.batch_decode(gen, skip_special_tokens=True))
