import imageio, os, torch, warnings, torchvision, argparse, json
from datasets import load_dataset
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from typing import Optional, List, Dict, Any, Set, Tuple
from pathlib import Path
import random

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat

class PhysicalEditingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str=None,
        num_frames: int = 81,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        max_pixels: int = 1920 * 1080,
        height: Optional[int] = None,
        width: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat: int = 1,
        key_frame_stride: int = 8,
        require_meta: bool = True,  # True: 无元信息就跳过；False: 允许空文本
        args=None,                  # 可选从 args 提取相同命名参数
    ):
        if args is not None:
            root_dir = getattr(args, "dataset_base_path", root_dir)
            num_frames = getattr(args, "num_frames", num_frames)
            height = getattr(args, "height", height)
            width = getattr(args, "width", width)
            max_pixels = getattr(args, "max_pixels", max_pixels)
            repeat = getattr(args, "dataset_repeat", repeat)

        self.root = Path(root_dir)
        self.num_frames = int(num_frames)
        self.time_division_factor = int(time_division_factor)
        self.time_division_remainder = int(time_division_remainder)
        self.max_pixels = int(max_pixels)
        self.height = height
        self.width = width
        self.height_division_factor = int(height_division_factor)
        self.width_division_factor = int(width_division_factor)
        self.repeat = int(repeat)
        self.require_meta = bool(require_meta)
        self.video_file_extension = video_file_extension
        self.key_frame_stride = int(key_frame_stride)

        if self.height is not None and self.width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif self.height is None and self.width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            print("One of height/width is None. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        # ------- 构建样本列表 -------
        self.samples: List[Dict[str, Any]] = self._build_samples(self.root)
        if len(self.samples) == 0:
            warnings.warn("PhysicalEditingDataset: no valid samples found.")

    # =========================
    # 封装的工具 / 读取函数
    # =========================
    def _is_video_file(self,p: Path) -> bool:
        return p.suffix.lower() in VIDEO_EXTS

    def _collect_leaf_dirs(self, root: Path) -> List[Path]:
        """
        一旦目录内出现视频文件，就把该目录视为叶子并停止向下。
        """
        leaf: List[Path] = []
        for cur, subdirs, files in os.walk(root):
            cur_p = Path(cur)
            vids_here = any(self._is_video_file(cur_p / f) for f in files)
            if vids_here:
                leaf.append(cur_p)
                subdirs[:] = []  # stop descending into children
        return sorted(set(leaf))

    # def _read_leaf_metadata(self, leaf: Path) -> Dict[int, Dict[str, str]]:
    #     """
    #     读取 leaf/metadata.json（数组），返回 idx -> {"prompt","state","transition"}
    #     """
    #     meta_path = leaf / "metadata.json"
    #     out: Dict[int, Dict[str, str]] = {}
    #     if not meta_path.exists():
    #         return out
    #     try:
    #         arr = json.loads(meta_path.read_text(encoding="utf-8"))
    #     except Exception:
    #         return out
    #     if not isinstance(arr, list):
    #         return out
    #     for obj in arr:
    #         try:
    #             idx = int(obj["idx"])
    #             out[idx] = {
    #                 "prompt": str(obj.get("prompt", "")),
    #                 "state": str(obj.get("State", "")),
    #                 "transition": str(obj.get("Transition", "")),
    #             }
    #         except Exception:
    #             continue
    #     return out

    def _read_leaf_metadata(self, leaf: Path) -> Dict[int, Dict[str, Any]]:
        uni_path = leaf / "unified_output_new_qwen.jsonl"
        out: Dict[int, Dict[str, Any]] = {}
        if not uni_path.exists():
            return out
        with uni_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                try:
                    idx = int(obj["idx"])
                except Exception:
                    continue
                out[idx] = obj
                # prompt = str(obj.get("prompt", ""))
                # edit_instruction = str(obj.get("edit_instruction", ""))
                # triplet = obj.get("triplet") or {}
                # state = str(obj.get("State", ""))
                # transition = str(obj.get("Transition", ""))
                # # 保障 triplet 字段存在且为 dict
                # if not isinstance(triplet, dict):
                #     triplet = {}
                # out[idx] = {
                #     "prompt": prompt,
                #     "edit_instruction": edit_instruction,
                #     "triplet": {
                #         "first_state_prompt": triplet.get("first_state_prompt", ""),
                #         "middle_transition_prompt": triplet.get("middle_transition_prompt", ""),
                #         "final_state_prompt": triplet.get("final_state_prompt", "")
                #     },
                #     "state": state,
                #     "transition": transition,
                # }
        return out

    def _read_filtered_names(self, leaf: Path) -> Set[str]:
        """
        读取 leaf/final_filter_videos.txt（若存在），每行文件名如 '35.mp4'
        """
        names = set()
        txt = leaf / "final_filter_videos.txt"
        if not txt.exists():
            return names
        for line in txt.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                names.add(name)
        return names

    def _list_videos(self, leaf: Path) -> List[Path]:
        return sorted([p for p in leaf.iterdir() if p.is_file() and self._is_video_file(p)])

    def read_high_rules(self, metainfo: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        stagea_principles = metainfo.get("stage_a", [])["principles"]
        high_rules = []
        for i, p in enumerate(stagea_principles):
            try:
                if str(p.get("priority","")).lower() != "high":
                    continue
                rid = str(p.get("id") or f"rule_{i}")
                instr = str(p.get("instruction","")).strip()
                cues = p.get("visual_cues", []) or []
                negs = p.get("negations", []) or []
                high_rules.append({
                    "id": rid,
                    "instruction": instr,
                    "visual_cues": [str(c).strip() for c in cues if str(c).strip()],
                    "negations": [str(n).strip() for n in negs if str(n).strip()]
                })
            except Exception:
                continue
        return high_rules
    
    def get_supported_and_contradicted_rules(self, metainfo: Dict[str, Any], high_rules: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        stageb_res = metainfo.get("stage_b", {})
        supported_rules, contradicted_rules = [], []
        rc_map = {rc.get("id",""): rc for rc in stageb_res.get("rule_checks", [])}
        for r in high_rules:
            rid = r.get("id","")
            rc = rc_map.get(rid, {})
            res = str(rc.get("result","unknown")).lower()
            if res == "supported":
                supported_rules.append({
                    "id": rid,
                    "instruction": r.get("instruction",""),
                    "matched_cues": rc.get("matched_cues", [])
                })
            elif res == "contradicted":
                contradicted_rules.append({
                    "id": rid,
                    "instruction": r.get("instruction","")
                })
        return supported_rules, contradicted_rules

    def _build_samples(self, root: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        leaf_dirs = self._collect_leaf_dirs(root)
        
        for leaf in leaf_dirs:
            meta_map = self._read_leaf_metadata(leaf)
            filtered = self._read_filtered_names(leaf)
            for vp in self._list_videos(leaf):
                if vp.name in filtered:
                    continue
                stem = vp.stem
                idx = int(stem) if stem.isdigit() else None
                if idx is None:
                    continue
                meta = meta_map.get(idx)
                if meta is None and self.require_meta:
                    # 没有元信息时跳过
                    continue
                if meta is None:
                    meta = {"prompt": "", "state": "", "transition": "", "edit_instruction": "", "triplet": {}}
                # breakpoint()
                high_rules = self.read_high_rules(meta)
                supported_rules, contradicted_rules = self.get_supported_and_contradicted_rules(meta, high_rules)

                samples.append({
                    "path": str(vp.resolve()),
                    "idx": idx,
                    "original_prompt": meta.get("prompt", ""),
                    "state": meta.get("state", ""),
                    "transition": meta.get("transition", ""),
                    "triplet": meta.get("triplet", {}), 
                    "prompt": meta.get("edit_instruction", ""),
                    "supported_rules": supported_rules,
                    "contradicted_rules": contradicted_rules,
                })
        # 按目录+idx 排序，稳定
        samples.sort(key=lambda x: (Path(x["path"]).parent.as_posix(), x["idx"]))
        print(f"[PhysicalEditingDataset] collected {len(samples)} samples from {len(leaf_dirs)} leaf dirs.")
        return samples

    # ---------- 与 baseline 对齐的图像工具 ----------
    def _crop_and_resize(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def _get_height_width(self, image: Image.Image) -> Tuple[int, int]:
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = max(self.height_division_factor,
                         (height // self.height_division_factor) * self.height_division_factor)
            width = max(self.width_division_factor,
                        (width // self.width_division_factor) * self.width_division_factor)
        else:
            height, width = self.height, self.width
        return height, width

    def _get_num_frames(self, reader) -> int:
        num_frames = self.num_frames
        try:
            total = int(reader.count_frames())
        except Exception:
            # 极少情况下 imageio 不支持 count_frames；退化为遍历
            total = 0
            try:
                while True:
                    reader.get_data(total)
                    total += 1
            except Exception:
                pass

        if total < num_frames:
            num_frames = total
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return max(1, num_frames)

    def _load_video(self, file_path: str) -> List[Image.Image]:
        try:
            reader = imageio.get_reader(file_path)
        except Exception as e:
            warnings.warn(f"cannot open video {file_path}: {e}")
            return []
        try:
            num_frames = self._get_num_frames(reader)
            frames: List[Image.Image] = []
            for frame_id in range(num_frames):
                try:
                    frame = reader.get_data(frame_id)
                except Exception:
                    break
                img = Image.fromarray(frame).convert("RGB")
                h, w = self._get_height_width(img)
                img = self._crop_and_resize(img, h, w)
                frames.append(img)
            reader.close()
        except Exception as e:
            reader.close()
            warnings.warn(f"error reading video {file_path}: {e}")
            return []
        return frames

    def extract_middle_key_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        # print(f"original frames: {len(frames)}")
        if len(frames) <= 2:
            return []
        middle_frames = frames[1:-1]
        stride = self.key_frame_stride
        selected_frames = []
        num = len(middle_frames)
        for i in range(0, num, stride):
            group = middle_frames[i:i+stride]
            if not group:
                continue
            mid_idx = len(group) // 2
            selected_frames.append(group[mid_idx])
        return selected_frames

    def stitch_middle_key_frames(self, frames: List[Image.Image]) -> Optional[Image.Image]:
        """
        3*2 concat middle frames to one image
        """
        if len(frames) != 6:
            warnings.warn(f"Expected 6 frames, but got {len(frames)}")
            return None

        # ensure all frames have the same size
        w, h = frames[0].size
        for img in frames:
            if img.size != (w, h):
                img = img.resize((w, h))
        
        stitched_image = Image.new("RGB", (2 * w, 3 * h))
        for idx, img in enumerate(frames):
            row = idx // 2
            col = idx % 2
            stitched_image.paste(img, (col * w, row * h))
        return stitched_image

    # =========================
    # Dataset 接口
    # =========================
    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, data_id: int) -> Optional[Dict[str, Any]]:
        rec = self.samples[data_id % len(self.samples)]
        video_path = rec["path"]
        frames = self._load_video(video_path)
        middle_key_frames = self.extract_middle_key_frames(frames)
        stitched_image = self.stitch_middle_key_frames(middle_key_frames)
        if not frames:
            warnings.warn(f"cannot load frames from {video_path}")
            return None

        sample = {
            # "video": frames,               # List[PIL.Image]
            "image": frames[-1],      # 最后一帧作为 gt image
            "edit_image": frames[0],  # 第一帧作为 edit image
            "middle_key_frames": middle_key_frames,
            "stitched_image": stitched_image,
            "prompt": rec["prompt"],
            "state": rec["state"],
            "transition": rec["transition"],
            "idx": rec["idx"],
            "path": video_path,
            "original_prompt": rec["original_prompt"],
            "triplet": rec["triplet"],
            "supported_rules": rec["supported_rules"],
            "contradicted_rules": rec["contradicted_rules"],
        }

        return sample


class Pica100kDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_id: str = "Andrew613/PICA-100K",
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_pixels: int = 1920 * 1080,
        height: Optional[int] = None,
        width: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        repeat: int = 1,
        args=None,
    ):
        if args is not None:
            dataset_id = getattr(args, "dataset_id", dataset_id)
            height = getattr(args, "height", height)
            width = getattr(args, "width", width)
            max_pixels = getattr(args, "max_pixels", max_pixels)
            repeat = getattr(args, "dataset_repeat", repeat)

        self.dataset_id = dataset_id
        self.split = split
        self.cache_dir = cache_dir
        self.max_pixels = int(max_pixels)
        self.height = height
        self.width = width
        self.height_division_factor = int(height_division_factor)
        self.width_division_factor = int(width_division_factor)
        self.repeat = int(repeat)

        if self.height is not None and self.width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif self.height is None and self.width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            print("One of height/width is None. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        self.data = load_dataset(self.dataset_id, split=self.split, cache_dir=self.cache_dir)

    def _crop_and_resize(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def _get_height_width(self, image: Image.Image) -> Tuple[int, int]:
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = max(self.height_division_factor,
                         (height // self.height_division_factor) * self.height_division_factor)
            width = max(self.width_division_factor,
                        (width // self.width_division_factor) * self.width_division_factor)
        else:
            height, width = self.height, self.width
        return height, width

    def _process_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        h, w = self._get_height_width(image)
        return self._crop_and_resize(image, h, w)

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def __getitem__(self, data_id: int) -> Optional[Dict[str, Any]]:
        rec = self.data[data_id % len(self.data)]
        src_img = rec.get("src_img")
        tgt_img = rec.get("tgt_img")
        if src_img is None or tgt_img is None:
            warnings.warn("Pica100kDataset: missing src_img/tgt_img.")
            return None
        prompt = rec.get("superficial_prompt", "")

        sample = {
            "image": self._process_image(tgt_img),
            "edit_image": [self._process_image(src_img)],
            "prompt": prompt,
        }
        return sample

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False, local_model_path=None, skip_download=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype, local_model_path=local_model_path, skip_download=skip_download) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name for logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name for logging")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="Save checkpoint every N steps (in addition to epoch-based saving)")
    parser.add_argument("--eval_every_n_steps", type=int, default=None, help="Evaluate model every N steps using the first sample from dataset")
    # resume options
    parser.add_argument("--resume_from", type=str, default=None, help="resume from: Accelerate saved state directory (state-*) or training weight file (.safetensors)")
    parser.add_argument("--resume_type", type=str, choices=["auto", "full", "model"], default="auto", help="auto: auto select by path type; full: resume from Accelerate state directory; model: only load model weight")
    parser.add_argument("--resume_original_num_processes", type=int, default=4, help="resume 时原训练使用的 GPU 数，缺少 metadata 时用于换卡推算")
    parser.add_argument("--local_model_path", type=str, default=None, help="Local directory containing the model files (avoids downloading from HuggingFace).")
    parser.add_argument("--dinov2_path", type=str, default=None, required=True, help="Path to the local DINOv2-with-registers-base model directory.")
    return parser
