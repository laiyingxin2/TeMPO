"""Minimal TeMPO demo: compress visual tokens of LLaVA-OneVision on one video.

Usage:
    python demo.py --video /path/to/video.mp4 --question "What happens in this video?" \
        --retain-ratio 0.10
"""

import argparse
import copy
import os
import warnings

import numpy as np
import torch
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frames = vr.get_batch(uniform_sampled_frames.tolist()).asnumpy()
    return frames  # (frames, height, width, channels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="path to a video file")
    parser.add_argument("--question", default="Please describe this video in detail.")
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--retain-ratio", type=float, default=0.10)
    parser.add_argument("--reference-impl", action="store_true", help="use the loop-form tempo_v1")
    args = parser.parse_args()

    os.environ["RETAIN_RATIO"] = str(args.retain_ratio)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from tempo import tempo, tempo_v1

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, "llava_qwen", device_map="auto", attn_implementation="sdpa", multimodal=True
    )
    model = (tempo_v1 if args.reference_impl else tempo)(model)
    model.eval()

    frames = load_video(args.video, args.max_frames)
    frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors = [frames]
    image_sizes = [frame.shape[:2] for frame in frames]

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{args.question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            modalities=["video"],
            do_sample=False,
            max_new_tokens=512,
        )
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
