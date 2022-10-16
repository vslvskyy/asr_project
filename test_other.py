import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_cer, calc_wer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        res_metrics = defaultdict(float)
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            for i in tqdm(range(len(batch["text"]))):
                res_metrics["n_obj"] += 1
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]

                target = batch["text"][i]
                pred_argmax = text_encoder.ctc_decode(argmax.cpu().numpy())

                probs = batch["log_probs"][i].cpu().numpy()
                pred_text_bs, pred_text_bs_lm = text_encoder.ctc_beam_search_lm(probs, beam_size=25)
                pred_text_bs, pred_text_bs_lm = pred_text_bs.lower(), pred_text_bs_lm.lower()

                cer_argmax, wer_argmax = calc_cer(target, pred_argmax), calc_wer(target, pred_argmax)
                res_metrics["cer_argmax"] += cer_argmax
                res_metrics["wer_argmax"] += wer_argmax

                cer_bs, wer_bs = calc_cer(target, pred_text_bs), calc_wer(target, pred_text_bs)
                res_metrics["cer_bs"] += cer_bs
                res_metrics["wer_bs"] += wer_bs

                cer_bs_lm, wer_bs_lm = calc_cer(target, pred_text_bs_lm), calc_wer(target, pred_text_bs_lm)
                res_metrics["cer_bs_lm"] += cer_bs_lm
                res_metrics["wer_bs_lm"] += wer_bs_lm

                results.append(
                    {
                        "ground_truth": target,

                        "pred_text_argmax": pred_argmax,
                        "cer argmax": cer_argmax,
                        "wer argmax": wer_argmax,

                        "pred_text_beam_search": pred_text_bs,
                        "cer beam search": cer_bs,
                        "wer beam search": wer_bs,

                        "pred_text_beam_search_lm": pred_text_bs_lm,
                        "cer beam search + lm": cer_bs_lm,
                        "wer beam search + lm": wer_bs_lm,
                    }
                )
        results.append(
            {
                "avg test-other cer argmax": res_metrics["cer_argmax"]/res_metrics["n_obj"],
                "avg test-other wer argmax": res_metrics["wer_argmax"]/res_metrics["n_obj"],
                "avg test-other cer beam search": res_metrics["cer_bs"]/res_metrics["n_obj"],
                "avg test-other wer beam search": res_metrics["wer_bs"]/res_metrics["n_obj"],
                "avg test-other cer beam search + lm": res_metrics["cer_bs_lm"]/res_metrics["n_obj"],
                "avg test-other wer beam search + lm": res_metrics["wer_bs_lm"]/res_metrics["n_obj"],
            }
        )
    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    config.config["data"] = {
        "test": {
            "datasets": [
                {
                    "type": "LibrispeechDataset",
                    "args": {
                        "part": "test-other"
                    }
                }
            ],
        }
    }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
