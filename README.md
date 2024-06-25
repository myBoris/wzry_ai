<p align="center">
    <a href="https://github.com/myBoris/wzry_ai">
        <img src="https://socialify.git.ci/myBoris/wzry_ai/image?description=1&font=Rokkitt&language=1&name=1&owner=1&theme=Auto" alt="wzry_ai"/>    
    </a>
</p>

<p align="center">
    <a href="https://github.com/myBoris/wzry_ai/stargazers">
        <img src="https://img.shields.io/github/stars/myBoris/wzry_ai?style=flat-square&label=STARS&color=%23dfb317" alt="stars">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/network/members">
        <img src="https://img.shields.io/github/forks/myBoris/wzry_ai?style=flat-square&label=FORKS&color=%2397ca00" alt="forks">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/issues">
        <img src="https://img.shields.io/github/issues/myBoris/wzry_ai?style=flat-square&label=ISSUES&color=%23007ec6" alt="issues">
    </a>
    <a href="https://github.com/myBoris/wzry_ai/pulls">
        <img src="https://img.shields.io/github/issues-pr/myBoris/wzry_ai?style=flat-square&label=PULLS&color=%23fe7d37" alt="pulls">
    </a>
</p>

---

## 访问统计

<p align="center">
    <a href="https://github.com/myBoris/wzry_ai">
        <img src="http://profile-counter.glitch.me/wzry_ai/count.svg" alt="count"/>
    </a>
</p>

---


# 声明:本项目的目的是为了学习人工智能，严禁外挂

# wzry_ai
人工智能模型玩王者荣耀。<br>本项目是开源项目，功能初步完成，后面还会更新。欢迎大家多多支持。<br>
### 未来规划
+ 继续更新本项目，并录制视频讲解
+ 实现物理上操作王者荣耀，就是制作一双仿生机器手，通过机器手来控制手机屏幕来操作

## 1.环境配置教程
+ 1.使用anaconda创建一个环境 (conda create --name wzry_ai python=3.10)
+ 2.安装必要的包,安装命令在 ([command.txt](command.txt)) 
+ + 这是必要的包
+ + `pip install -r requirements.txt`
+ + 这是pytorch和cuda的安装
+ + `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
+ + onnxruntime-gpu的安装
+ + `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
+ +  onnxruntime-gpu的运行时如果出现下面问题<br> `Could not locate zlibwapi.dll. Please make sure it is in your library path!`
+ + 解决方法:<br> 
    `复制下面文件夹的文件: (2022.4.2这个可能不一样，按照你自己系统就行，Nsight Systems这个是一样的)

        C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll

    复制到这个文件夹，并且改名为: zlibwapi:

        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll`

## 2.训练教程<br>
####   将[start.onnx](src%2Fstart.onnx)模型放在[src](src)目录下，直接运行[train.py](train.py)，会生成模型([wzry_ai.pt](src%2Fwzry_ai.pt))

## 3.使用教程<br>
#### 直接运行[testModel.py](testModel.py)就行，模型([wzry_ai.pt](src%2Fwzry_ai.pt))放在[src](src)目录下，以后直接更新这个模型文件即可

## 4.在真机和模拟器上的配置<br>
### 真机和模拟器按键位置的修改,我使用的是真机(2400 x 1080),屏幕大小没关系<br>
+ #### 1.这个文件([androidController.py](androidController.py))的position就是坐标点，改这里会影响点击的位置.<br>
![屏幕截图1.png](https://github.com/myBoris/wzry_ai/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE1.png)
+ #### 如何获取点击的位置，我这里写了一个脚本，更改坐标后，运行一下，就知道点击的位置在哪.里面直接输入坐标实时更新还有问题，以后再修改，重新运行就行<br>
![屏幕截图2.png](https://github.com/myBoris/wzry_ai/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE2.png)


## 5.其他
    qq交流群:687853827

## 6.广告
+ [【腾讯云】特惠活动
1 元开启 GPU 炼丹之旅
澎湃算力，即开即用，使用高性能GPU服务HAI，快速部署LLM、AI绘画等应用，助你玩转AIGC！](https://cloud.tencent.com/act/cps/redirect?redirect=36749&cps_key=11812351d85cc069a0941ce4c8d07693)

+ [【腾讯云】2核2G4M云服务器新老同享99元/年，续费同价](https://cloud.tencent.com/act/cps/redirect?redirect=5990&cps_key=11812351d85cc069a0941ce4c8d07693&from=console)

+ [【阿里云】云服务器 精选特惠，老用户升级最低享6.5折，协助您选择最合适配置方案。](https://www.aliyun.com/product/ecs?userCode=cgwj31jh)


