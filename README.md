# 声明:本项目的目的是为了学习人工智能，严禁外挂

# wzry_ai
人工智能模型玩王者荣耀。<br>本项目是开源项目，功能初步完成，后面还会更新。欢迎大家多多支持。<br>我未来会录制视频讲解，敬请期待

## 1.环境配置教程
+ 1.使用anaconda创建一个环境 (conda create --name wzry_ai python=3.10)
+ 2.安装必要的包,安装命令在 ([command.txt](command.txt)) 
+ + 这是必要的包
+ + `pip install -r requirements.txt`
+ + 这是pytorch和cuda的安装
+ + `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## 2.使用教程<br>
#### 直接运行[testModel.py](testModel.py)就行，模型([wzry_ai.pt](src%2Fwzry_ai.pt))放在[src](src)目录下，以后直接更新这个模型文件即可

## 3.在真机和模拟器上的配置<br>
### 真机和模拟器按键位置的修改<br>
+ #### 1.这个文件([androidController.py](androidController.py))的position就是坐标点，改这里会影响点击的位置.<br>
![屏幕截图1.png](https://github.com/myBoris/wzry_ai/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE1.png)
+ #### 如何获取点击的位置，我这里写了一个脚本，更改坐标后，运行一下，就知道点击的位置在哪.里面直接输入坐标实时更新还有问题，以后再修改，重新运行就行<br>
![屏幕截图2.png](https://github.com/myBoris/wzry_ai/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE2.png)


## 4.其他
    qq交流群:687853827

## 5.广告
+ [特惠活动
1 元开启 GPU 炼丹之旅
澎湃算力，即开即用，使用高性能GPU服务HAI，快速部署LLM、AI绘画等应用，助你玩转AIGC！](https://cloud.tencent.com/act/cps/redirect?redirect=36749&cps_key=11812351d85cc069a0941ce4c8d07693) <br><br>

+ [【腾讯云】2核2G4M云服务器新老同享99元/年，续费同价](https://cloud.tencent.com/act/cps/redirect?redirect=5990&cps_key=11812351d85cc069a0941ce4c8d07693&from=console)




