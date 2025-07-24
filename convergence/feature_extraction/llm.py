import gc
import os


from tqdm import trange
from tqdm import tqdm

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from ..nsd import load_dataset

from ..models import get_models

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoConfig,
)
from dmf.alerts import send_message


from .. import utils


def extract_llm_features(
    filenames,
    output_dir,
    dataset,
    dataset_name,
    texts,
    subset,
    pool,
    force_remake,
    force_download,
    batch_size,
    prompt="prompt",
    caption_idx=0,
    qlora=False,
    from_init=False,
):
    # dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    # texts = [str(x["text"][args.caption_idx]) for x in dataset]

    for llm_model_name in (pbar := tqdm(filenames, position=0)):
        pbar.set_description(f"Processing {llm_model_name}")
        send_message(f"Processing {llm_model_name}")
        try:
            save_path = utils.to_feature_filename(
                output_dir,
                dataset_name,
                subset,
                llm_model_name,
                pool=pool,
                prompt=prompt,
                caption_idx=caption_idx,
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"\ndataset: \t{dataset_name}")
            print(f"subset:    \t{subset}")
            print(f"processing:\t{llm_model_name}")
            print(f"save_path: \t{save_path}")

            if os.path.exists(save_path) and not force_remake:
                print("file exists. skipping")
                continue

            language_model = load_llm(
                llm_model_name, qlora=qlora, force_download=force_download, from_init=from_init
            )
            llm_param_count = sum([p.numel() for p in language_model.parameters()])
            tokenizer = load_tokenizer(llm_model_name)

            tokens = tokenizer(texts, padding="longest", return_tensors="pt")
            llm_feats, losses, bpb_losses = [], [], []

            # hack to get around HF mapping data incorrectly when using model-parallel
            device = next(language_model.parameters()).device

            for i in trange(0, len(dataset), batch_size):
                # get embedding cuda device
                token_inputs = {
                    k: v[i : i + batch_size].to(device).long()
                    for (k, v) in tokens.items()
                }

                with torch.no_grad():
                    if "olmo" in llm_model_name.lower():
                        llm_output = language_model(
                            input_ids=token_inputs["input_ids"],
                            attention_mask=token_inputs["attention_mask"],
                            output_hidden_states=True,
                        )
                    else:
                        llm_output = language_model(
                            input_ids=token_inputs["input_ids"],
                            attention_mask=token_inputs["attention_mask"],
                        )

                    loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                    losses.extend(avg_loss.cpu())

                    bpb = utils.cross_entropy_to_bits_per_unit(
                        loss.cpu(), texts[i : i + batch_size], unit="byte"
                    )
                    bpb_losses.extend(bpb)

                    # make sure to do all the processing in cpu to avoid memory problems
                    if pool == "avg":
                        feats = torch.stack(llm_output["hidden_states"]).permute(
                            1, 0, 2, 3
                        )
                        mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                        feats = (feats * mask).sum(2) / mask.sum(2)
                    elif pool == "last":
                        feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                        feats = torch.stack(feats).permute(1, 0, 2)
                    else:
                        raise NotImplementedError(f"unknown pooling {pool}")
                    llm_feats.append(feats.cpu())

            print(f"average loss:\t{torch.stack(losses).mean().item()}")
            save_dict = {
                "feats": torch.cat(llm_feats).cpu(),
                "num_params": llm_param_count,
                "mask": tokens["attention_mask"].cpu(),
                "loss": torch.stack(losses).mean(),
                "bpb": torch.stack(bpb_losses).mean(),
            }

            torch.save(save_dict, save_path)

            del language_model, tokenizer, llm_feats, llm_output
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error processing {llm_model_name}: {e}")
            send_message(f"Error rocessing {llm_model_name}")

        send_message(f"Finish Processing {llm_model_name}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return


def auto_determine_dtype():
    """automatic dtype setting. override this if you want to force a specific dtype"""
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def check_bfloat16_support():
    """checks if cuda driver/device supports bfloat16 computation"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(current_device)
        if compute_capability[0] >= 7:  # Check if device supports bfloat16
            return True
        else:
            return False
    else:
        return None


def load_llm(llm_model_path, qlora=False, force_download=False, from_init=False):
    """load huggingface language model"""
    compute_dtype, torch_dtype = auto_determine_dtype()

    quantization_config = None
    if qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if from_init:
        config = AutoConfig.from_pretrained(
            llm_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            force_download=force_download,
            output_hidden_states=True,
        )
        print(config)
        language_model = AutoModelForCausalLM.from_config(config)
        language_model = language_model.to(torch_dtype)
        language_model.init_weights()
        language_model = language_model.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        language_model = language_model.eval()
    else:
        language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            force_download=force_download,
            output_hidden_states=True,
        ).eval()

    return language_model


def load_tokenizer(llm_model_path):
    """setting up tokenizer. if your tokenizer needs special settings edit here."""
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    if "huggyllama" in llm_model_path:
        tokenizer.pad_token = "[PAD]"
    else:
        # pass
        # tokenizer.add_special_tokens({"pad_token":"<pad>"})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer
