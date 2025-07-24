"""
caption_dataset_pixtral.py

Script to generate image captions using the Pixtral-12B vision-language model.

This script processes a dataset of images (NSD, THINGS, or BOLD5000), encodes them in base64,
and sends them to a Pixtral model via vLLM's multi-modal chat interface using predefined or custom prompts.
It returns one caption per image and stores the output in a CSV or Parquet file.

Usage:
    python caption_dataset_pixtral.py --dataset <nsd|things|bold5000> \
                                      --output <captions.csv|.parquet> \
                                      [--prompt <caption|motion|indoor|scene|custom_prompt>]

Arguments:
    --dataset: Dataset name (must be one of: nsd, things, bold5000).
    --output: Path to output file (.csv or .parquet).
    --prompt: Caption prompt to use. Defaults to "caption".
              Can be a predefined key (`caption`, `motion`, `indoor`, `scene`) or a custom string.

Predefined Prompts:
    - caption: One-sentence image description.
    - motion: Binary classification into 'motion' or 'static'.
    - indoor: Scene type classification: 'indoor', 'natural', 'urban', or 'object'.
    - scene: Scene-object dominance classification: 'Scene-dominated', 'Object-in-Scene', or 'Object-dominated'.

Requirements:
    - Images must be accessible from disk and listed in the dataset's `image` column.
    - The dataset must be loadable via the `load_dataset` function from the respective module.
    - Requires `vllm`, `torch`, and Pixtral-12B model support.

Outputs:
    A CSV or Parquet file with:
        - All original dataset columns (excluding the image data itself),
        - A new `caption` column,
        - Metadata columns: `model` and `prompt`.

"""

from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
from dmf.models import get_device
import pandas as pd
import logging
from tqdm import trange
from dmf.alerts import alert
from pathlib import Path
import base64
import argparse


IMAGE_COLUMN = "image"
MODEL_NAME = "mistralai/Pixtral-12B-2409"
BATCH_SIZE = 8
MAX_TOKENS = 8192
MAX_MODEL_LEN = 12288

PREDEFINED_PROMPTS = {
    "caption": "Describe this image in one sentence.",
    "motion": "Analyze the image to determine whether it depicts motion or is static. Motion is defined as the presence or strong implication of movement (e.g., people walking, animals running, vehicles in action). If no movement is present or implied, classify it as static. Reply only with the single word 'motion' or 'static'.",
    "indoor": "Classify each image into exactly one of these four categories: 'indoor' (interior spaces, rooms, furniture), outdoor natural: 'natural' (natural landscapes, animals in nature), outdoor urban: 'urban' (cityscapes, streets, vehicles, urban infrastructure), or 'object' (isolated items or food without clear scene context). Reply only with the name of the class and no other text. Only one of the words: indoor, natural, urban or object",
    "scene": "Classify images by visual dominance into these categories: 'Scene-dominated' (rich context/background, minimal dominating person/animal/object), 'Object-in-Scene' (clear foreground person/animal/object dominating the image, with visible but secondary background), or 'Object-dominated' (isolated foreground person/animal/object, minimal/neutral background context). Only reply with the term: 'Scene-dominated', 'Object-in-Scene', 'Object-dominated'.",
}


def image_to_base64(image_path):
    path = Path(image_path)
    file_extension = (
        path.suffix.lower()
    )  # Get the extension and convert it to lowercase

    # Map file extensions to MIME types
    if file_extension in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif file_extension == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(
            "Unsupported image format. Only .jpg, .jpeg, and .png are supported."
        )

    # Read the image and convert to base64
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Return the base64 string with the correct MIME type
    return f"data:{mime_type};base64,{base64_string}"


def load_llm(model_name):
    # Load the LLM model
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
    llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=MAX_MODEL_LEN)
    return llm, sampling_params


def caption_batch(images, llm, sampling_params, prompt) -> str:
    captions = []

    for image_path in images:
        # Convert local image to base64
        base64_image = image_to_base64(image_path)

        # Prepare the message with the base64-encoded image and prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }
        ]
        # Generate the caption using the LLM
        try:
            outputs = llm.chat(messages, sampling_params=sampling_params)
            caption = outputs[0].outputs[0].text.strip()
            captions.append(caption)
        except Exception as e:
            logger.error(f"Error generating caption for image: {image_path}")
            logger.error(e)
            captions.append("")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return captions


def caption_dataset(dataset, llm, sampling_params, prompt, batch_size=None):
    batch_size = batch_size or BATCH_SIZE
    captions = []

    for i in trange(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size][IMAGE_COLUMN]
        captions += caption_batch(batch, llm, sampling_params, prompt=prompt)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return captions


@alert(output=True)
def main():
    # Dataset arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["nsd", "things", "bold5000"]
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="caption")

    args = parser.parse_args()

    assert args.output.endswith(".csv") or args.output.endswith(
        ".parquet"
    ), "Output file must be a .csv or .parquet file"

    prompt = args.prompt
    if prompt in PREDEFINED_PROMPTS:
        logger.info(f"Using predefined prompt: {prompt}")
        prompt = PREDEFINED_PROMPTS[prompt]

    logger.info(f"Using prompt: {prompt}")

    # Load dataset
    if args.dataset == "bold5000":
        from convergence.bold5000 import load_dataset
    elif args.dataset == "nsd":
        from convergence.nsd import load_dataset
    elif args.dataset == "things":
        from convergence.things import load_dataset

    dataset = load_dataset(cast_column=False)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset loaded: {args.dataset}. Samples: {len(dataset)}")

    # Load the LLM model
    llm, sampling_params = load_llm(MODEL_NAME)
    logger.info(f"Model loaded: {MODEL_NAME}")
    captions = caption_dataset(
        dataset, llm, prompt=prompt, sampling_params=sampling_params
    )

    output_data = {}
    for feature in dataset.column_names:
        if feature != IMAGE_COLUMN:
            output_data[feature] = dataset[feature]

    output_data["caption"] = captions
    output_df = pd.DataFrame(output_data)
    output_df["model"] = MODEL_NAME
    output_df["prompt"] = prompt
    output_file = Path(args.output)

    # If ends with .parquet save as parquet, otherwise save as csv
    if output_file.suffix == ".parquet":
        output_df.to_parquet(output_file, index=False)
    else:
        output_df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("caption")
    main()
