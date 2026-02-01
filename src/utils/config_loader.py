# 配置加载器 (Configuration Loader)
# 用于加载YAML配置文件，提供统一的配置管理接口
# 本模块实现了一个配置加载系统，可以从指定目录读取YAML格式的配置文件
# 并提供缓存机制避免重复读取相同的配置文件

import os  # 导入操作系统模块，用于处理文件路径
import yaml  # 导入YAML解析库，用于解析YAML格式的配置文件
from typing import Dict, Any  # 导入类型提示，用于函数参数和返回值的类型注解


class ConfigLoader:
    """
    配置加载器类，用于加载和管理YAML配置文件
    """
    
    def __init__(self, config_dir: str):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录的绝对路径，所有配置文件都将从该目录下加载
        
        初始化过程：
        1. 保存配置文件目录路径
        2. 创建空字典用于缓存已加载的配置，避免重复读取文件
        """
        self.config_dir = config_dir  # 存储配置文件目录的路径
        self.configs = {}  # 初始化空字典，用于缓存已加载的配置，键为配置名，值为配置内容
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载指定的配置文件
        
        Args:
            config_name: 配置文件名称（不含扩展名），例如"paths"对应"paths.yaml"文件
            
        Returns:
            配置字典: 包含从YAML文件解析出的所有配置项的字典
            
        Raises:
            FileNotFoundError: 当指定的配置文件不存在时抛出此异常
            
        工作流程：
        1. 首先检查配置是否已缓存，如已缓存则直接返回缓存的配置
        2. 如未缓存，构建完整的配置文件路径
        3. 检查文件是否存在，不存在则抛出异常
        4. 打开并读取配置文件，使用yaml.safe_load解析YAML内容
        5. 将解析后的配置存入缓存并返回
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")  # 构建完整的配置文件路径
        
        if config_name in self.configs:  # 检查配置是否已缓存
            return self.configs[config_name]  # 如已缓存，直接返回缓存的配置，避免重复读取文件
        
        if not os.path.exists(config_path):  # 检查配置文件是否存在
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")  # 文件不存在时抛出异常
        
        with open(config_path, 'r', encoding='utf-8') as f:  # 打开配置文件，指定UTF-8编码
            config = yaml.safe_load(f)  # 使用yaml库的safe_load方法解析YAML内容为Python字典
        
        self.configs[config_name] = config  # 将解析后的配置存入缓存字典
        return config  # 返回解析后的配置字典
    
    def get_paths_config(self) -> Dict[str, Any]:
        """
        获取路径配置
        
        Returns:
            路径配置字典: 包含项目中各种路径设置的字典，如数据路径、输出路径等
            
        说明：
        这是一个便捷方法，相当于调用load_config("paths")，专门用于加载paths.yaml文件
        paths.yaml通常包含项目中使用的各种文件和目录的路径配置
        """
        return self.load_config("paths")  # 调用通用的load_config方法加载名为"paths"的配置文件
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典: 包含机器学习模型相关参数的字典，如模型类型、超参数等
            
        说明：
        这是一个便捷方法，相当于调用load_config("model_config")，专门用于加载model_config.yaml文件
        model_config.yaml通常包含机器学习模型的各种配置参数，如模型类型、学习率、批量大小等
        """
        return self.load_config("model_config")  # 调用通用的load_config方法加载名为"model_config"的配置文件
    
    def get_variables_config(self) -> Dict[str, Any]:
        """
        获取变量配置
        
        Returns:
            变量配置字典: 包含项目中使用的各种变量设置的字典，如特征名称、阈值等
            
        说明：
        这是一个便捷方法，相当于调用load_config("variables")，专门用于加载variables.yaml文件
        variables.yaml通常包含项目中使用的各种变量设置，如特征名称列表、阈值设置、常量定义等
        """
        return self.load_config("variables")  # 调用通用的load_config方法加载名为"variables"的配置文件
    
    def get_viz_config(self) -> Dict[str, Any]:
        """
        获取可视化配置
        
        Returns:
            可视化配置字典: 包含数据可视化相关设置的字典，如图表类型、颜色方案、标签等
            
        说明：
        这是一个便捷方法，相当于调用load_config("viz_config")，专门用于加载viz_config.yaml文件
        viz_config.yaml通常包含数据可视化的各种配置参数，如图表类型、颜色方案、标签设置等
        """
        return self.load_config("viz_config")  # 调用通用的load_config方法加载名为"viz_config"的配置文件


# 创建配置加载器实例的工厂函数
def create_config_loader(config_dir: str = None) -> ConfigLoader:
    """
    创建配置加载器实例的工厂函数
    
    Args:
        config_dir: 配置文件目录路径，如果为None，则自动定位到项目根目录下的configs目录
        
    Returns:
        ConfigLoader实例: 一个已初始化的配置加载器对象，可直接用于加载配置
        
    工作流程：
    1. 如果未指定config_dir参数：
       a. 获取当前模块文件(config_loader.py)所在的目录
       b. 基于当前目录，计算项目根目录下configs文件夹的绝对路径
          (假设目录结构为: 项目根目录/src/utils/config_loader.py，配置目录为: 项目根目录/configs/)
    2. 使用确定的配置目录路径创建并返回ConfigLoader实例
    
    使用示例：
    ```python
    # 使用默认配置目录
    config_loader = create_config_loader()
    paths_config = config_loader.get_paths_config()
    
    # 或指定自定义配置目录
    custom_loader = create_config_loader("/path/to/custom/configs")
    ```
    """
    if config_dir is None:  # 如果未指定配置目录
        # 获取当前文件(config_loader.py)所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 从当前文件位置(src/utils/)向上导航两级，然后进入configs目录
        # 即: 项目根目录/configs/
        config_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'configs'))
    
    # 创建并返回ConfigLoader实例
    return ConfigLoader(config_dir)