# 修改自: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501


class Registry():
    """
    注册表类，用于提供 名称 -> 对象的映射，支持第三方用户的自定义模块。

    创建一个注册表 (例如 backbone 注册表):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    注册一个对象:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    或者:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        参数:
            name (str): 注册表的名称
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"名为 '{name}' 的对象已经注册在 "
                                             f"'{self._name}' 注册表中!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        在 `obj.__name__` 下注册给定的对象。
        既可以用作装饰器，也可以不作为装饰器使用。
        使用方法请参见此类的文档字符串。
        """
        if obj is None:
            # 用作装饰器
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # 用作函数调用
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        """
        根据名称获取注册的对象。
        """
        ret = self._obj_map.get(name)
   
        if ret is None:
            raise KeyError(f"在 '{self._name}' 注册表中找不到名为 '{name}' 的对象!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


# 预定义各种模块的注册表
DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
