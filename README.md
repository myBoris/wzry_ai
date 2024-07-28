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


>声明:本项目的目的是为了学习人工智能，严禁外挂

### 一、项目简介
:::success

- 这是一个开源的人工智能模型玩王者荣耀的项目。
- 同时第一期想做的工程全部完成了，以后将在这个基础上进行升级

:::
<br>

### 二、环境配置教程
:::info

- 1.下载anaconda并安装<br>
   下载地址:
   `
    https://www.anaconda.com/download
   `<br><br>
- 2.使用anaconda创建一个环境<br>
    命令: `
      conda create --name wzry_ai python=3.10
   `<br><br>
- 3.激活这个环境<br>
    命令: `
      conda activate wzry_ai
   `<br><br>
- 4.在wzry_ai环境安装必要的包<br>
  1. 执行下面环境安装命令<br>
     `
         pip install -r requirements.txt
     `<br><br>
  2. 执行下面环境命令pytorch和cuda
  
     ` 
         pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     `<br><br>
  3. 执行下面环境命令安装onnxruntime-gpu<br>
     如果cuda是11

      `
          pip install onnxruntime-gpu
     `<br>
     如果cuda是12<br>
     
     `
      pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
      `<br><br>
  4. onnxruntime-gpu的运行时如果出现下面问题
  
     `
         Could not locate zlibwapi.dll. Please make sure it is in your library path!
     `<br>
     解决方法:<br> 
     复制下面文件夹的文件: (2022.4.2这个可能不一样，按照你自己系统就行，Nsight Systems这个是一样的)<br>
  
        `C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll`

     复制到这个文件夹，并且改名为: zlibwapi:<br> 
        `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll`<br><br>
:::
<br>

### 三、训练教程
:::warning

- 将qq群文件下载的onnx模型放在models目录下，直接运行train.py，会生成模型(wzry_ai.pt)

:::
<br>

### 四、在真机和模拟器上的配置(坐标工具类暂时失效，等待重写)
:::danger

- 修改按键映射，按键映射在这个文件里[argparses.py](argparses.py)
:::
<br>

### 五、其他
:::success

- qq交流群:687853827

:::
<br>

### 六、广告
:::success

- AI创业做啥好呢？
- [【腾讯云】特惠活动
1 元开启 GPU 炼丹之旅
澎湃算力，即开即用，使用高性能GPU服务HAI，快速部署LLM、AI绘画等应用，助你玩转AIGC！](https://cloud.tencent.com/act/cps/redirect?redirect=36749&cps_key=11812351d85cc069a0941ce4c8d07693)

- [【腾讯云】2核2G4M云服务器新老同享99元/年，续费同价](https://cloud.tencent.com/act/cps/redirect?redirect=5990&cps_key=11812351d85cc069a0941ce4c8d07693&from=console)

- [【阿里云】云服务器 精选特惠，老用户升级最低享6.5折，协助您选择最合适配置方案。](https://www.aliyun.com/product/ecs?userCode=cgwj31jh)


:::
<br>

#### 注：开源不易，共同努力。