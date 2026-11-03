import math
import os
import random
import re
from abc import ABC

import torch
import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image
from method_wrappers.binding import bind_llava_methods


def allocate_proportional_budgets(total_budget, scores, min_budget=1):
    num_slots = len(scores)
    if num_slots == 0:
        return []

    total_budget = max(int(total_budget), num_slots * min_budget)
    if total_budget == num_slots * min_budget:
        return [min_budget] * num_slots

    clean_scores = []
    for score in scores:
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        clean_scores.append(max(0.0, value))

    score_sum = sum(clean_scores)
    if score_sum <= 0:
        clean_scores = [1.0] * num_slots
        score_sum = float(num_slots)

    extra_budget = total_budget - num_slots * min_budget
    raw_extra = [extra_budget * score / score_sum for score in clean_scores]
    floored_extra = [int(math.floor(value)) for value in raw_extra]
    budgets = [min_budget + extra for extra in floored_extra]

    remaining = total_budget - sum(budgets)
    if remaining > 0:
        order = sorted(
            range(num_slots),
            key=lambda idx: (raw_extra[idx] - floored_extra[idx], clean_scores[idx], -idx),
            reverse=True,
        )
        for idx in order[:remaining]:
            budgets[idx] += 1

    return budgets


def safe_linear_solve(matrix, rhs):
    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError:
        try:
            return torch.linalg.lstsq(matrix, rhs).solution
        except RuntimeError:
            return torch.linalg.pinv(matrix) @ rhs


class BaseVideoMethodMixin(ABC):
    def _format_nonvideo_image_features(
        self,
        image_feature,
        image_idx,
        image_sizes,
        mm_patch_merge_type,
        image_aspect_ratio,
    ):
        if mm_patch_merge_type == "flat":
            return image_feature.flatten(0, 1)

        if not mm_patch_merge_type.startswith("spatial"):
            raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")

        if image_feature.shape[0] > 1:
            base_image_feature = image_feature[0]
            image_feature = image_feature[1:]
            height = width = self.get_vision_tower().num_patches_per_side
            assert height * width == base_image_feature.shape[0]

            matched_anyres_max_num_patches = None
            if "anyres_max" in image_aspect_ratio:
                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                if matched_anyres_max_num_patches:
                    max_num_patches = int(matched_anyres_max_num_patches.group(1))

            if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                if not hasattr(self.get_vision_tower(), "image_size"):
                    raise ValueError("vision_tower_image_size is not found in the vision tower.")
                vision_tower_image_size = self.get_vision_tower().image_size
                if image_sizes is None:
                    raise ValueError("image_sizes is required for anyres image formatting.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            else:
                image_feature = image_feature.view(2, 2, height, width, -1)

            if "maxpool2x2" in mm_patch_merge_type:
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = nn.functional.max_pool2d(image_feature, 2)
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                unit = image_feature.shape[2]
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                c, h, w = image_feature.shape
                times = math.sqrt(h * w / (max_num_patches * unit**2))
                if times > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature,
                        [int(h // times), int(w // times)],
                        mode="bilinear",
                    )[0]
                image_feature = torch.cat(
                    (
                        image_feature,
                        self.get_model().image_newline[:, None, None]
                        .expand(*image_feature.shape[:-1], 1)
                        .to(image_feature.device),
                    ),
                    dim=-1,
                )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            elif "unpad" in mm_patch_merge_type:
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                image_feature = torch.cat(
                    (
                        image_feature,
                        self.get_model().image_newline[:, None, None]
                        .expand(*image_feature.shape[:-1], 1)
                        .to(image_feature.device),
                    ),
                    dim=-1,
                )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            else:
                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                image_feature = image_feature.flatten(0, 3)

            if "nobase" not in mm_patch_merge_type:
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            return image_feature

        image_feature = image_feature[0]
        if "unpad" in mm_patch_merge_type:
            image_feature = torch.cat((image_feature, self.get_model().image_newline[None].to(image_feature.device)), dim=0)
        return image_feature

    @staticmethod
    def _trim_known_text_query_suffix(segment, token_ids):
        if token_ids is None or token_ids.numel() == 0 or segment.numel() == 0:
            return segment

        token_list = token_ids.tolist()
        cut_end = len(token_list)

        # Qwen chatml generation prompt:
        # <|im_end|>\n<|im_start|>assistant\n
        known_suffixes = (
            (151645, 198, 151644, 77091, 198),
        )
        token_tail = tuple(token_list)
        for suffix in known_suffixes:
            if len(token_tail) >= len(suffix) and token_tail[-len(suffix):] == suffix:
                cut_end -= len(suffix)
                token_list = token_list[:cut_end]
                break

        # Multiple-choice prompts often append option blocks beginning with
        # "\nA." / "\nB." ... . For text-guided wrappers we want the question
        # body, not the answer candidates.
        option_label_ids = {32, 33, 34, 35, 36}  # "A"..."E" in Qwen
        newline_marker_ids = {198, 624}  # "\n" and ".\n"
        cut_at = None
        for idx in range(1, len(token_list) - 1):
            if token_list[idx] in option_label_ids and token_list[idx + 1] == 13 and token_list[idx - 1] in newline_marker_ids:
                cut_at = idx
                break

        # If no option block was found, still try to remove explicit answer
        # instructions that begin on a new line.
        if cut_at is None:
            answer_prefix_ids = {16141}  # "Answer" in Qwen
            for idx in range(1, len(token_list)):
                if token_list[idx] in answer_prefix_ids and token_list[idx - 1] in newline_marker_ids:
                    cut_at = idx
                    break

        if cut_at is not None and cut_at > 8:
            cut_end = min(cut_end, cut_at)

        trimmed = segment[:cut_end]
        return trimmed if trimmed.numel() > 0 else segment

    @staticmethod
    def _select_text_query_tokens(text_segments, token_id_segments=None):
        if not text_segments:
            raise ValueError("Expected at least one text segment when selecting text query tokens.")

        non_empty_segments = [segment for segment in text_segments if segment.numel() > 0]
        if not non_empty_segments:
            return text_segments[0]

        mode = os.environ.get("TEXT_QUERY_SELECTION", "auto").strip().lower()
        if mode in ("concat", "all"):
            return torch.cat(non_empty_segments, dim=0)

        if mode == "longest":
            longest_idx = max(range(len(non_empty_segments)), key=lambda idx: (non_empty_segments[idx].shape[0], idx))
            return non_empty_segments[longest_idx]

        if mode == "auto" and len(non_empty_segments) > 1:
            # Chat templates usually inject a short system/role span before `<image>`.
            # In auto mode, prefer the main post-image user-query span so wrappers
            # that consume `text_tokens` are not polluted by role headers or answer
            # option blocks. Callers that truly need the concatenated text can still
            # opt back in via TEXT_QUERY_SELECTION=concat.
            non_empty_pairs = []
            for idx, segment in enumerate(text_segments):
                if segment.numel() == 0:
                    continue
                token_ids = None if token_id_segments is None else token_id_segments[idx]
                non_empty_pairs.append((segment, token_ids))
            longest_idx = max(range(len(non_empty_pairs)), key=lambda idx: (non_empty_pairs[idx][0].shape[0], idx))
            segment, token_ids = non_empty_pairs[longest_idx]
            return BaseVideoMethodMixin._trim_known_text_query_suffix(segment, token_ids)

        return torch.cat(non_empty_segments, dim=0)

    @staticmethod
    def _unpack_vision_tower_output(vision_outputs):
        if isinstance(vision_outputs, torch.Tensor):
            return vision_outputs, None, None, vision_outputs.dtype

        if isinstance(vision_outputs, (tuple, list)):
            if len(vision_outputs) == 0:
                raise ValueError("Vision tower returned an empty output sequence.")

            image_features = vision_outputs[0]
            attn_weights = vision_outputs[1] if len(vision_outputs) > 1 else None
            metric = vision_outputs[2] if len(vision_outputs) > 2 else None
            images_dtype = vision_outputs[3] if len(vision_outputs) > 3 else getattr(image_features, "dtype", None)
            return image_features, attn_weights, metric, images_dtype

        return vision_outputs, None, None, getattr(vision_outputs, "dtype", None)

    def encode_images(self, images):
        vision_outputs = self.get_model().get_vision_tower()(images)
        image_features, _, _, _ = BaseVideoMethodMixin._unpack_vision_tower_output(vision_outputs)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_multi(self, images):
        vision_outputs = self.get_model().get_vision_tower()(images)
        image_features, attn_weights, metric, images_dtype = BaseVideoMethodMixin._unpack_vision_tower_output(vision_outputs)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_weights, metric, images_dtype

    def _ensure_method_module(self):
        raise NotImplementedError

    def _compress_video_tokens(self, feats, cls_attn, key_metric, text_tokens, nextline_token):
        raise NotImplementedError

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=['image'],
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        mm_newline_position = getattr(self.config, 'mm_newline_position', 'one_token')

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = [idx for idx, modality in enumerate(modalities) if modality == 'video']
            images_list = [image if image.ndim == 4 else image.unsqueeze(0) for image in images]
            concat_images = torch.cat(images_list, dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, attn_weights, metric, _ = self.encode_images_multi(concat_images)
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if mm_newline_position != 'grid':
                nextline_token = torch.tensor([198], device=encoded_image_features.device)
                nextline_token = self.get_model().embed_tokens(nextline_token)
            else:
                nextline_token = self.model.image_newline[None].to(encoded_image_features.device)

            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            metric_split = torch.split(metric, split_sizes) if metric is not None else [None] * len(split_sizes)
            attn_split = torch.split(attn_weights, split_sizes) if attn_weights is not None else [None] * len(split_sizes)

            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    pooled_feat = self.get_2dPool(image_feat)
                    pooled_key = self.get_2dPool(metric_split[idx]) if metric_split[idx] is not None else None
                    pooled_attn = self.get_2dPool(attn_split[idx].unsqueeze(-1)).squeeze(-1) if attn_split[idx] is not None else None
                    image_features.append((pooled_feat, pooled_attn, pooled_key))
                else:
                    image_features.append(
                        self._format_nonvideo_image_features(
                            image_feat,
                            idx,
                            image_sizes,
                            mm_patch_merge_type,
                            image_aspect_ratio,
                        )
                    )
        else:
            image_features = self.encode_images(images)
            video_idx_in_batch = []
            nextline_token = None

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for idx in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[idx] + 1:image_token_indices[idx + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[idx] + 1:image_token_indices[idx + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            sample_text_tokens = BaseVideoMethodMixin._select_text_query_tokens(
                list(cur_input_embeds_no_im),
                list(cur_input_ids_noim),
            )

            for idx in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[idx])
                cur_new_labels.append(cur_labels_noim[idx])
                if idx < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    if batch_idx in video_idx_in_batch and isinstance(cur_image_features, tuple):
                        pooled_feat, pooled_attn, pooled_key = cur_image_features
                        cur_image_features = torch.cat(
                            self._compress_video_tokens(pooled_feat, pooled_attn, pooled_key, sample_text_tokens, nextline_token),
                            dim=0,
                        )
                    elif batch_idx in video_idx_in_batch and isinstance(cur_image_features, list):
                        cur_image_features = torch.cat(cur_image_features, dim=0)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for batch_idx, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == 'left':
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed,
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[batch_idx, -cur_len:] = cur_new_labels
                    attention_mask[batch_idx, -cur_len:] = True
                    position_ids[batch_idx, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[batch_idx, :cur_len] = cur_new_labels
                    attention_mask[batch_idx, :cur_len] = True
                    position_ids[batch_idx, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if getattr(self.config, 'use_pos_skipping', False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
