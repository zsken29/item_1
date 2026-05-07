"""
Python装饰器用法演示
循序渐进地介绍装饰器的各种用法
"""

# 导入必要的模块
import time
import functools

# 第1步：最简单的装饰器
def simple_decorator(func):
    """最简单的装饰器"""
    def wrapper():
        print("装饰器开始执行")
        func()
        print("装饰器结束执行")
    return wrapper

@simple_decorator
def hello():
    print("Hello, World!")

# 第2步：带参数的装饰器
def decorator_with_args(func):
    """带参数的装饰器"""
    def wrapper(*args, **kwargs):
        print("装饰器开始执行")
        result = func(*args, **kwargs)
        print("装饰器结束执行")
        return result
    return wrapper

@decorator_with_args
def greet(name):
    print(f"Hello, {name}!")
    return f"Greeted {name}"

# 第3步：带返回值的装饰器
def decorator_with_return(func):
    """带返回值的装饰器"""
    def wrapper(*args, **kwargs):
        print("装饰器开始执行")
        result = func(*args, **kwargs)
        print(f"函数返回值: {result}")
        print("装饰器结束执行")
        return result
    return wrapper

@decorator_with_return
def add(a, b):
    return a + b

# 第4步：带参数的装饰器函数
def timer(units="秒"):
    """带参数的装饰器函数"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            duration = end - start
            if units == "毫秒":
                duration *= 1000
            print(f"函数执行时间: {duration:.2f} {units}")
            return result
        return wrapper
    return decorator

@timer()  # 使用默认参数

def slow_function():
    time.sleep(0.5)
    print("函数执行完成")

@timer(units="毫秒")  # 指定参数

def another_slow_function():
    time.sleep(0.3)
    print("另一个函数执行完成")

# 第5步：使用functools.wraps保留函数元信息
def decorator_with_wraps(func):
    """使用functools.wraps的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器开始执行")
        result = func(*args, **kwargs)
        print("装饰器结束执行")
        return result
    return wrapper

@decorator_with_wraps
def function_with_metadata():
    """这是一个有元数据的函数"""
    print("函数执行")

# 第6步：多个装饰器
def decorator1(func):
    def wrapper(*args, **kwargs):
        print("装饰器1开始")
        result = func(*args, **kwargs)
        print("装饰器1结束")
        return result
    return wrapper

def decorator2(func):
    def wrapper(*args, **kwargs):
        print("装饰器2开始")
        result = func(*args, **kwargs)
        print("装饰器2结束")
        return result
    return wrapper

@decorator1
@decorator2
def function_with_multiple_decorators():
    print("函数执行")

# 第7步：类装饰器
class ClassDecorator:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        print("类装饰器开始执行")
        result = self.func(*args, **kwargs)
        print("类装饰器结束执行")
        return result

@ClassDecorator
def function_with_class_decorator():
    print("函数执行")

# 第8步：带参数的类装饰器
class ClassDecoratorWithArgs:
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f"{self.prefix}: 装饰器开始执行")
            result = func(*args, **kwargs)
            print(f"{self.prefix}: 装饰器结束执行")
            return result
        return wrapper

@ClassDecoratorWithArgs("INFO")
def function_with_class_decorator_args():
    print("函数执行")

# 测试所有装饰器
def test_all_decorators():
    print("\n=== 测试最简单的装饰器 ===")
    hello()
    
    print("\n=== 测试带参数的装饰器 ===")
    result = greet("Alice")
    print(f"greet函数返回: {result}")
    
    print("\n=== 测试带返回值的装饰器 ===")
    sum_result = add(10, 20)
    print(f"add函数返回: {sum_result}")
    
    print("\n=== 测试带参数的装饰器函数 ===")
    slow_function()
    another_slow_function()
    
    print("\n=== 测试使用functools.wraps的装饰器 ===")
    function_with_metadata()
    print(f"函数名: {function_with_metadata.__name__}")
    print(f"函数文档: {function_with_metadata.__doc__}")
    
    print("\n=== 测试多个装饰器 ===")
    function_with_multiple_decorators()
    
    print("\n=== 测试类装饰器 ===")
    function_with_class_decorator()
    
    print("\n=== 测试带参数的类装饰器 ===")
    function_with_class_decorator_args()

if __name__ == "__main__":
    test_all_decorators()