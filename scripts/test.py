from pathlib import Path

import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Pick a frame where SAM already detected a bus.
IMAGE_PATH = (
    "data/frames/test_bus/"
    "frame_000003.jpg"
)


# ==========================================
# Resolve image path
# ==========================================

image_path = Path(
    IMAGE_PATH
).resolve()

print("\n======================================")
print("IMAGE TEST")
print("======================================")

print("Image path:")
print(image_path)

print("\nExists:")
print(image_path.exists())

if not image_path.exists():
    raise FileNotFoundError(
        f"Image does not exist:\n{image_path}"
    )


# IMPORTANT:
#
# Do NOT use:
#
# image_path.as_uri()
#
# On Windows that creates:
#
# file:///C:/Users/...
#
# qwen_vl_utils can have problems with the
# resulting URL-encoded Windows path.
#
# Instead give it a normal absolute path:
#
# C:\Users\...\frame_000003.jpg

image_path_string = str(
    image_path
)

print("\nPath being given to Qwen:")
print(image_path_string)


# ==========================================
# Load Qwen
# ==========================================

print("\n======================================")
print("LOADING QWEN")
print("======================================")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

processor = (
    AutoProcessor.from_pretrained(
        MODEL_NAME
    )
)

if device == "cuda":

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    )

else:

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    )

    model = model.to(
        "cpu"
    )

model.eval()


# ==========================================
# Create message
# ==========================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",

                # Normal Windows path.
                "image":
                    image_path_string,
            },
            {
                "type": "text",
                "text": (
                    "Look carefully at this image. "
                    "Describe what is visibly present. "
                    "Then explicitly state whether "
                    "a bus is visible."
                ),
            },
        ],
    }
]


# ==========================================
# Build Qwen chat prompt
# ==========================================

text = (
    processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
)


# ==========================================
# Process image
# ==========================================

print("\n======================================")
print("PROCESSING VISION INPUT")
print("======================================")

(
    image_inputs,
    video_inputs,
) = process_vision_info(
    messages
)


print(
    "image_inputs:",
    type(image_inputs),
)

print(
    "number of image inputs:",
    (
        len(image_inputs)
        if image_inputs is not None
        else 0
    ),
)

print(
    "video_inputs:",
    video_inputs,
)


# ==========================================
# Build model inputs
# ==========================================

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


if device == "cuda":

    inputs = inputs.to(
        model.device
    )

else:

    inputs = inputs.to(
        "cpu"
    )


# ==========================================
# Generate response
# ==========================================

print("\n======================================")
print("GENERATING RESPONSE")
print("======================================")

with torch.inference_mode():

    generated_ids = (
        model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
    )


# ==========================================
# Remove original prompt tokens
# ==========================================

generated_ids_trimmed = [

    output_ids[
        len(input_ids):
    ]

    for (
        input_ids,
        output_ids,
    ) in zip(
        inputs.input_ids,
        generated_ids,
    )
]


# ==========================================
# Decode Qwen response
# ==========================================

response = (
    processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
)


# ==========================================
# Print result
# ==========================================

print("\n======================================")
print("QWEN RESPONSE")
print("======================================")

print(response)

print("\n======================================")