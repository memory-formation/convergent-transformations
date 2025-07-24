import os
import gc

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import trange, tqdm
import torch
from dmf.alerts import send_message
from torchvision.models.feature_extraction import get_graph_node_names

from .. import utils


def extract_lvm_features(
    filenames,
    dataset,
    pool="avg",
    output_dir=".",
    dataset_name="nsd",
    subset="all",
    force_remake: bool = False,
    batch_size=64,
    pretrained=True,
):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argpafrom torchvision.models.feature_extraction import get_graph_node_namesrse arguments
    """
    assert pool == "cls", "pooling is not supported for lvm features"

    for lvm_model_name in (pbar := tqdm(filenames)):
        send_message(f"Processing {lvm_model_name}")
        pbar.set_description(f"Processing {lvm_model_name}")
        # assert "vit" in lvm_model_name, "only vision transformers are supported"

        save_path = utils.to_feature_filename(
            output_dir,
            dataset_name,
            subset,
            lvm_model_name,
            pool=pool,
            prompt=None,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path) and not force_remake:
            continue
        try:
            vision_model = (
                timm.create_model(lvm_model_name, pretrained=pretrained).cuda().eval()
            )
            lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

            transform = create_transform(
                **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
            )

            if "vit" in lvm_model_name:
                #names = get_graph_node_names(vision_model)
                #print(names)
                return_nodes = [
                    f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
                ]
            elif "resnet" in lvm_model_name:
                raise NotImplementedError("resnet not implemented")

            else:
                raise NotImplementedError(f"unknown model {lvm_model_name}")

            vision_model = create_feature_extractor(
                vision_model, return_nodes=return_nodes
            )
            lvm_feats = []

            for i in trange(0, len(dataset), batch_size):
                with torch.no_grad():

                    ims = torch.stack(
                        [transform(im) for im in dataset[i : i + batch_size]["image"]]
                    ).cuda()
                    lvm_output = vision_model(ims)

                    if pool == "cls":
                        feats = [v[:, 0, :] for v in lvm_output.values()]
                        feats = torch.stack(feats).permute(1, 0, 2)

                    lvm_feats.append(feats.cpu())

            torch.save(
                {"feats": torch.cat(lvm_feats), "num_params": lvm_param_count},
                save_path,
            )

            del vision_model, transform, lvm_feats, lvm_output
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error processing {lvm_model_name}: {e}")
            send_message(f"Error Processing {lvm_model_name}")

        send_message(f"Finish processing {lvm_model_name}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
