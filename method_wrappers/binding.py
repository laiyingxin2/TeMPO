"""Utilities for safely binding method wrapper hooks onto LLaVA models."""

from types import MethodType

CORE_METHOD_NAMES = (
    "prepare_inputs_labels_for_multimodal",
    "encode_images",
    "encode_images_multi",
    "_ensure_method_module",
    "_compress_video_tokens",
)


def _bind_instance_attr(dst_obj, attr_name, attr_value):
    if isinstance(attr_value, staticmethod):
        setattr(dst_obj, attr_name, attr_value.__func__)
    elif isinstance(attr_value, classmethod):
        setattr(dst_obj, attr_name, attr_value.__get__(dst_obj.__class__, dst_obj.__class__))
    elif callable(attr_value):
        descriptor = getattr(attr_value, "__get__", None)
        if descriptor is not None:
            setattr(dst_obj, attr_name, descriptor(dst_obj, dst_obj.__class__))
        else:
            setattr(dst_obj, attr_name, MethodType(attr_value, dst_obj))


def _bind_class_attr(dst_cls, attr_name, attr_value):
    if isinstance(attr_value, staticmethod):
        setattr(dst_cls, attr_name, staticmethod(attr_value.__func__))
    elif isinstance(attr_value, classmethod):
        setattr(dst_cls, attr_name, classmethod(attr_value.__func__))
    elif callable(attr_value):
        setattr(dst_cls, attr_name, attr_value)


def _bind_attr(dst, attr_name, attr_value):
    if isinstance(dst, type):
        _bind_class_attr(dst, attr_name, attr_value)
    else:
        _bind_instance_attr(dst, attr_name, attr_value)


def bind_llava_methods(dst, src_cls):
    """Copy core hooks plus inherited helper methods onto a class or instance."""

    for attr_name in CORE_METHOD_NAMES:
        if hasattr(src_cls, attr_name):
            _bind_attr(dst, attr_name, getattr(src_cls, attr_name))

    # Walk the full MRO so helper methods inherited from shared mixins, such as
    # EnhancedVideoMethodMixin, are also copied onto the destination class.
    for cls in reversed(src_cls.mro()):
        if cls is object:
            continue

        for attr_name, attr_value in cls.__dict__.items():
            if attr_name in CORE_METHOD_NAMES or attr_name.startswith("__"):
                continue

            _bind_attr(dst, attr_name, attr_value)
