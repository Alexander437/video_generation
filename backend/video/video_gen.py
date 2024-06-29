VIDEO_GEN_REGISTRY = {}


def register_video_gen(gen_name: str, cls):
    global VIDEO_GEN_REGISTRY
    if gen_name not in VIDEO_GEN_REGISTRY:
        VIDEO_GEN_REGISTRY[gen_name] = cls


def get_video_gen(gen_name: str):
    if gen_name not in VIDEO_GEN_REGISTRY:
        raise ValueError(f"No video generator registered with name {gen_name}")
    return VIDEO_GEN_REGISTRY[gen_name]


def list_video_gen():
    return list(VIDEO_GEN_REGISTRY.keys())
