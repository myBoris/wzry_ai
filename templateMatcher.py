import glob
import os
import time

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class TemplateMatcher:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.template_paths = {
            'successes_0': 'templateImages/template_success0.jpg',
            'successes_1': 'templateImages/template_success1.jpg',
            'successes_2': 'templateImages/template_success2.jpg',
            'successes_4': 'templateImages/template_success4.jpg',
            'failed_0': 'templateImages/template_failure0.jpg',
            'failed_1': 'templateImages/template_failure1.jpg',
            'failed_2': 'templateImages/template_failure2.jpg',
            'death_0': 'templateImages/template_death0.jpg',
            'death_1': 'templateImages/template_death1.jpg',
            'death_2': 'templateImages/template_death2.jpg',
            'started_red': 'templateImages/template_start_red.jpg',
            'started_blue': 'templateImages/template_start_blue.jpg',
            'started_blue2': 'templateImages/template_start_blue2.jpg'
        }
        self.templates = self._load_templates()

    def _load_templates(self):
        templates = {}
        for template_type, template_path in self.template_paths.items():
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            # template = cv2.resize(template, (template.shape[1], template.shape[0]))
            if template is not None:
                templates[template_type] = template
            else:
                print(f"无法从路径 {template_path} 读取模板图像，跳过此模板。")
        return templates

    def _match_template(self, image_data, template):

        # 灰度转换
        image_data_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # 执行模板匹配
        result = cv2.matchTemplate(image_data_gray, template, cv2.TM_CCOEFF_NORMED)

        # 找到结果中匹配度大于等于阈值的部分
        match_locations = np.where(result >= self.threshold)
        # 计算匹配点的数量
        match_count = len(match_locations[0])
        # 总共可能的匹配点数量
        total_possible_matches = result.shape[0] * result.shape[1]
        # 计算匹配比例
        match_ratio = match_count / total_possible_matches
        return match_ratio


    def match(self, image_data: np.ndarray) -> str:
        best_match_type = None
        best_match_ratio = 0

        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._match_template, image_data, template): template_type
                for template_type, template in self.templates.items()
            }

            # 等待所有线程完成
            for future in as_completed(futures):
                match_ratio = future.result()
                template_type = futures[future]
                # 更新最佳匹配
                if match_ratio > best_match_ratio:
                    best_match_ratio = match_ratio
                    best_match_type = template_type.split("_")[0]

        return best_match_type


def find_images(folder_path):
    # 定义支持的图片扩展名
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]

    # 初始化一个空列表来存储找到的图片路径
    image_files = []

    # 遍历所有扩展名，并添加匹配的文件到列表中
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))

    return image_files
# 使用示例
if __name__ == "__main__":
    # 创建模板匹配器对象
    matcher = TemplateMatcher(threshold=0.8)


    # 查找文件夹中的所有图片
    images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\01_started")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\03_successes")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\04_failed")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\06_death")
    # 遍历每个图片并调用 template_match 方法
    for image_path in images:
        # 读取图像数据
        image_data = cv2.imread(image_path)

        # 记录开始时间
        start_time = time.time()

        # 调用模板匹配方法
        result = matcher.match(image_data)

        # 记录并打印这一步骤的时间
        step1_time = time.time()
        print(f"运行时间: {step1_time - start_time:.3f} 秒")

        # 打印结果
        print(f"Image: {image_path} - {result}")
