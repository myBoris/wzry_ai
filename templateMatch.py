import cv2
import os
import numpy as np


class TemplateMatcher:
    def __init__(self, template_dir, threshold=0.8):
        """
        初始化 TemplateMatcher 类，使用模板目录和匹配阈值。
        """
        self.templates = []
        self.template_names = []
        self.threshold = threshold
        self.load_templates(template_dir)

    def load_templates(self, template_dir):
        """
        从指定目录加载模板，并将其存储为灰度图像。
        """
        for template_file in os.listdir(template_dir):
            template_path = os.path.join(template_dir, template_file)
            template = cv2.imread(template_path, 0)  # 确保模板以灰度图读取
            if template is not None:
                self.templates.append(template)
                self.template_names.append(os.path.splitext(template_file)[0])

    def crop_to_circle(self, image):
        """
        将输入图像裁剪为以 (x, y) 为中心，半径为 r 的圆形区域。
        """
        x, y, r = 745, 20, 15
        height, width = image.shape[:2]

        # 创建掩膜
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # 将掩膜应用到图像上
        result = cv2.bitwise_and(image, image, mask=mask)

        # 将图像裁剪到圆形的边界框
        cropped_image = result[y - r:y + r, x - r:x + r]

        return cropped_image

    def multi_scale_template_matching(self, image, template, scales):
        """
        执行多尺度模板匹配，返回最佳匹配分数和位置。
        """
        best_match_score = -1
        best_location = None

        for scale in scales:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if image.shape[0] < resized_template.shape[0] or image.shape[1] < resized_template.shape[1]:
                continue
            res = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_match_score:
                best_match_score = max_val
                best_location = max_loc

        return best_match_score, best_location

    def match_template(self, image):
        """
        使用加载的模板匹配输入图像，并返回最佳匹配的模板名称。
        """
        # 将图像裁剪到感兴趣的圆形区域
        cropped_image = self.crop_to_circle(image)

        # 确保裁剪后的图像为灰度图
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 4:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2GRAY)
        elif len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # 初始化最佳匹配结果
        best_match_score = -1
        best_template_name = None

        # 对每个模板执行模板匹配
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # 可调整的尺度范围
        for template, template_name in zip(self.templates, self.template_names):
            match_score, _ = self.multi_scale_template_matching(cropped_image, template, scales)
            if match_score > best_match_score:
                best_match_score = match_score
                best_template_name = template_name

        # 如果分数高于阈值，则返回最佳匹配的模板名称
        if best_match_score >= self.threshold:
            return best_template_name
        return None


# 使用示例
if __name__ == "__main__":
    template_dir = "template"
    image_path = "H:\\video\\monster\\train\\attackMonster_bear\\video_video_frame_0014313.jpg"
    matcher = TemplateMatcher(template_dir)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        matched_template = matcher.match_template(image)
        print(f"匹配的模板: {matched_template}")
    else:
        print("加载图像失败。")
