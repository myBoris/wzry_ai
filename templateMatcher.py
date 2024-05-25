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
            'successes': 'templateImages/success',
            'failed': 'templateImages/failure',
            'death': 'templateImages/death'
        }

        self.start_template_paths = {
            'started': 'templateImages/start'
        }
        self.templates = self._load_templates_from_dirs(self.template_paths)
        self.start_templates = self._load_templates_from_dirs(self.start_template_paths)

    @staticmethod
    def _load_templates_from_dirs(template_dirs):
        templates = {}
        for template_type, dir_path in template_dirs.items():
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            key = f"{template_type}_{file_name}"
                            templates[key] = template
                        else:
                            print(f"无法从路径 {file_path} 读取模板图像，跳过此模板。")
            else:
                print(f"路径 {dir_path} 不存在。")
        return templates

    def _match_template(self, image, template):
        # 灰度转换
        image_data_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    def is_back_home(self, image, min_area=3500, max_area=4500):
        h, w = image.shape[:2]

        # 计算裁剪区域
        x_start = int(w * 0.3)
        y_start = int(h * 0.6)
        crop_width = int(w * 0.4)
        crop_height = int(h * 0.2)
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        # 裁剪图像
        cropped_image = image[y_start:y_end, x_start:x_end]

        # 转换为灰度图像
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 进行边缘检测
        edged = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 计算轮廓的周长
            peri = cv2.arcLength(contour, True)
            # 使用多边形逼近轮廓
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # 如果多边形有四个顶点，我们认为它是一个矩形
            if len(approx) == 4:
                # 获取矩形的边界框
                x, y, w, h = cv2.boundingRect(approx)

                # 计算矩形的面积
                area = w * h

                # 判断矩形面积是否在指定范围内
                if min_area <= area <= max_area and 10 <= h <= 20 and 250 <= w <= 265:
                    return "backHome"

        return None

    def match(self, image) -> str:
        best_match_type = None
        best_match_ratio = 0

        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._match_template, image, template): template_type
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

        if best_match_type is None:
            best_match_type = self.is_back_home(image)


        return best_match_type

    def match_start(self, image) -> str:
        best_match_type = None
        best_match_ratio = 0

        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._match_template, image, template): template_type
                for template_type, template in self.start_templates.items()
            }

            # 等待所有线程完成
            for future in as_completed(futures):
                match_ratio = future.result()
                template_type = futures[future]
                # print(template_type,match_ratio)
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
    # # 创建模板匹配器对象
    matcher = TemplateMatcher(threshold=0.8)
    #
    #
    # # 读取图像数据
    # image_data = cv2.imread("H:\\AI\\work\\wzry_status\\video\\video6\\frame_0021470.jpg")
    #
    #
    # # 记录开始时间
    # start_time = time.time()
    #
    # cv2.imshow("image_data", image_data)
    #
    # # 调用模板匹配方法
    # result = matcher.match_start(image_data)
    #
    # # 记录并打印这一步骤的时间
    # step1_time = time.time()
    # print(f"运行时间: {step1_time - start_time:.3f} 秒")
    #
    # # 打印结果
    # print(f"Image:  {result}")
    #
    #
    # cv2.waitKey()

    # 查找文件夹中的所有图片
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\01_started")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\03_successes")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\04_failed")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\06_death")
    # images = find_images("H:\\AI\\work\\wzry_status\\srcImages\\09_attackSmallDragon")
    images = find_images("tmp")
    # 遍历每个图片并调用 template_match 方法
    for image_path in images:
        # 读取图像数据
        image_data = cv2.imread(image_path)

        # 记录开始时间
        start_time = time.time()

        # 调用模板匹配方法
        # result = matcher.match_start(image_data)
        result = matcher.match(image_data)

        # 记录并打印这一步骤的时间
        step1_time = time.time()
        print(f"运行时间: {step1_time - start_time:.3f} 秒")

        # 打印结果
        print(f"Image: {image_path} - {result}")
