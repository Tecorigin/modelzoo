## docker run参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
-d |指定容器运行于前台还是后台|-|
-i | 打开STDIN，用于控制台交互|-|
-t | 支持终端登录|-|
-name | 指定容器名称 | -name r50_pt |
-ipc |设置容器的IPC命名空间| -ipc=host |
-p |用于将容器内部的端口映射到主机上的端口| -p host_port:docker_port
-v| 用于将主机上的目录或文件挂载到容器内部 | -v host_path:docker_path
--device | 用于指定驱动设备 | --device /dev/tcaicard0
--shm-size | 用于设置容器内 /dev/shm 目录的大小 | --shm-size=128g
--network | 设置容器的网络空间，--network=host时，容器将共享主机的 IP 地址、网络接口和端口 | --network=host