# wzry_ai
人工智能模型玩王者荣耀

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
    qq群:687853827
    请开发者喝杯水
![屏幕截图3.png](images%2F%C6%C1%C4%BB%BD%D8%CD%BC3.png)