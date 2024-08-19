from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

# class Depth_Anything_Preprocessor:
#     @classmethod
#     def INPUT_TYPES(s):
#         return define_preprocessor_inputs(
#             ckpt_name=INPUT.COMBO(
#                 ["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"]
#             ),
#             resolution=INPUT.RESOLUTION()
#         )

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "execute"

#     CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

#     def execute(self, image, ckpt_name="depth_anything_vitl14.pth", resolution=512, **kwargs):
#         from custom_controlnet_aux.depth_anything import DepthAnythingDetector

#         model = DepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
#         out = common_annotator_call(model, image, resolution=resolution)
#         del model
#         return (out, )

class Depth_Anything_Preprocessor:
    model_cache = {}  # 添加模型缓存

    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            ckpt_name=INPUT.COMBO(
                ["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"]
            ),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, ckpt_name="depth_anything_vitb14.pth", resolution=512, use_half_precision=True, **kwargs):
        from custom_controlnet_aux.depth_anything import DepthAnythingDetector

        # 如果模型未加载，则加载并缓存
        if ckpt_name not in self.model_cache:
            model = DepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
            if use_half_precision:
                model = model.half()  # 使用半精度
            self.model_cache[ckpt_name] = model
        else:
            model = self.model_cache[ckpt_name]

        out = common_annotator_call(model, image, resolution=resolution)
        return (out, )

# class Zoe_Depth_Anything_Preprocessor:
#     @classmethod
#     def INPUT_TYPES(s):
#         return define_preprocessor_inputs(
#             environment=INPUT.COMBO(["indoor", "outdoor"]),
#             resolution=INPUT.RESOLUTION()
#         )

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "execute"

#     CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

#     def execute(self, image, environment="indoor", resolution=512, **kwargs):
#         from custom_controlnet_aux.zoe import ZoeDepthAnythingDetector
#         ckpt_name = "depth_anything_metric_depth_indoor.pt" if environment == "indoor" else "depth_anything_metric_depth_outdoor.pt"
#         model = ZoeDepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
#         out = common_annotator_call(model, image, resolution=resolution)
#         del model
#         return (out, )

class Zoe_Depth_Anything_Preprocessor:
    model_cache = {}  # 添加模型缓存

    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            environment=INPUT.COMBO(["indoor", "outdoor"]),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, environment="indoor", resolution=512, use_half_precision=True, **kwargs):
        from custom_controlnet_aux.zoe import ZoeDepthAnythingDetector
        ckpt_name = "depth_anything_metric_depth_indoor.pt" if environment == "indoor" else "depth_anything_metric_depth_outdoor.pt"

        # 如果模型未加载，则加载并缓存
        if ckpt_name not in self.model_cache:
            model = ZoeDepthAnythingDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
            if use_half_precision:
                model = model.half()  # 使用半精度
            self.model_cache[ckpt_name] = model
        else:
            model = self.model_cache[ckpt_name]

        out = common_annotator_call(model, image, resolution=resolution)
        return (out, )


NODE_CLASS_MAPPINGS = {
    "DepthAnythingPreprocessor": Depth_Anything_Preprocessor,
    "Zoe_DepthAnythingPreprocessor": Zoe_Depth_Anything_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingPreprocessor": "Depth Anything",
    "Zoe_DepthAnythingPreprocessor": "Zoe Depth Anything"
}