import onnxruntime as ort
import numpy as np
import cv2


class OnnxRunner:
    def __init__(self, model_path, input_width=640, input_height=640, confidence_thres=0.5, iou_thres=0.4, classes=[]):
        """
        :param model_path: ONNX 模型文件的路径
        :param input_width: 模型输入的宽度
        :param input_height: 模型输入的高度
        :param confidence_thres: 过滤检测结果的置信度阈值
        :param iou_thres: 非极大值抑制的 IOU 阈值
        :param classes: 类别名称列表
        """
        self.img_height = 0
        self.img_width = 0
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = classes

    def preprocess(self, image):
        """
        预处理输入图像

        :param image: OpenCV 读取的图像
        :return: 预处理后的图像和原始图像尺寸
        """
        self.img_height, self.img_width = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))  # CHW
        image_expanded = np.expand_dims(image_transposed, axis=0)  # NCHW
        return image_expanded

    def postprocess(self, outputs):
        """
        后处理模型输出

        :param outputs: 模型输出
        :return: 过滤后的检测框
        """
        outputs = np.squeeze(outputs[0])
        # print(outputs.shape)

        # 获取输出数组的行数
        rows = outputs.shape[0]

        # 存储检测到的边界框、得分和类别ID的列表
        detections = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组中的每一行
        for i in range(rows):
            # print(int(outputs[i][0]), int(outputs[i][1]), int(outputs[i][2]), int(outputs[i][3]), round(float(outputs[i][4]), 4), int(outputs[i][5]))

            # print(round(float(outputs[i][4].item()), 4))

            # 从当前行中提取类别得分
            classes_scores = round(float(outputs[i][4]), 4)

            # 如果最大得分高于置信度阈值
            if classes_scores >= self.confidence_thres:
                # 获取得分最高的类别ID
                class_id = int(outputs[i][5])

                # 从当前行中提取边界框坐标
                x1, y1, x2, y2 = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int(x1 * x_factor)
                top = int(y1 * y_factor)
                width = int(x2 * x_factor)
                height = int(y2 * y_factor)

                # 添加检测信息到列表中
                detections.append({
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "score": classes_scores,
                    "box": [left, top, width, height]
                })

        # 应用非极大值抑制过滤重叠的边界框
        # indices = cv2.dnn.NMSBoxes(
        #     [det["box"] for det in detections],
        #     [det["score"] for det in detections],
        #     self.confidence_thres,
        #     self.iou_thres
        # )
        #
        # #
        # # # 根据非极大值抑制后的索引过滤检测结果
        # final_detections = [detections[i] for i in indices]

        return detections

    def run(self, image):
        """
        运行模型推理

        :param image: OpenCV 读取的图像
        :return: 过滤后的检测框
        """
        input_data = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        return self.postprocess(outputs)

    def get_max_label(self, image):
        # 运行推理
        detections = self.run(image)

        max_score = 0.0
        result = None
        for det in detections:
            score = float(det["score"])
            if score > max_score:
                max_score = score
                result = det["class_name"]
        return result

    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果

        :param image: OpenCV 读取的图像
        :param detections: 检测结果列表
        :return: 带有检测框的图像
        """
        for det in detections:
            left, top, right, bottom = det["box"]

            # 绘制边界框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # 绘制类别和置信度
            label = f'{det["class_name"]}: {det["score"]:.2f}'
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


# 使用示例
if __name__ == "__main__":
    # 使用示例
    model_path = 'models/start.onnx'
    classes = ["started"]  # 添加实际的类别名称

    # model_path = 'yolov10b.models'
    # classes = [
    #             "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    #             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    #             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    #             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    #             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    #             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    #             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    #             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    #             "scissors", "teddy bear", "hair drier", "toothbrush"
    #         ]

    runner = OnnxRunner(model_path, classes=classes)

    # 读取图像
    image_path = "H:\\AI\\work\\wzry_status\\srcImages\\06_death\\frame_0000005.jpg"
    # image_path = "ultralytics/assets/zidane.jpg"
    # image_path = "ultralytics/assets/frame_0002598.jpg"
    image = cv2.imread(image_path)

    print(runner.get_max_label(image))

    # 运行推理
    detections = runner.run(image)
    print("检测结果:", detections)

    # 在图像上绘制检测结果
    image_with_detections = runner.draw_detections(image, detections)

    # 显示图像
    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
