import cv2
import torch

if __name__ == "__main__":

    # 加载 YOLO 模型
    # model = YOLO(model_path)
    model = torch.hub.load("train/reword/yolov9-main/", 'custom', path='best.pt', force_reload=True, source='local')
    device = torch.device("cuda")
    # model.to(device)
    conf = 0.2  # 最小置信度
    iou = 0.5   # 最小交并比

    # 实例化视频对象
    cap = cv2.VideoCapture("../dataset/Record_2.mp4")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # 如果帧读取正确，ret 为 True
        if not ret:
            print('无法收到视频帧数据（该视频流是否已结束？），程序正在退出')
            break

        if count < 5:
            count = count + 1
            continue
        else:
            count = 0

        # 将帧转为模型输入所需格式
        # frame_tensor = torch.from_numpy(frame).to(device).float()
        # frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW

        # 进行对象检测
        results = model(frame)

        # 解析结果
        for detection in results.xyxy[0]:  # results.xyxy[0] 是检测结果张量
            x1, y1, x2, y2, conf, class_id = detection.cpu().numpy()
            print(f"ID: {int(class_id)}, Conf: {conf:.2f}, Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

        # 显示该帧
        cv2.imshow('frame0', frame)

        # 当按下键盘 q 时，退出程序
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源和关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
