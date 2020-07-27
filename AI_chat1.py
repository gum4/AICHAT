#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:28:35 2020

@author: gumenghan
"""


import os
import sqlite3
import torch
import weakref, collections
from functools import wraps
import time
import aiml
import sys
import torchtext.vocab as vocab
from collections import defaultdict



def download_glove(cache_dir):
    glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
    return glove

class LocalCache( ):
    notFound = object()

    # list dict不支持弱引用，必须使用子类
    class Dict(dict):
        #析构函数，必须定义
        def __del__(self):
            pass

    def __init__(self,MAX_len):
        #当对像的引用只剩弱引用时， garbage collection 可以销毁引用并将其内存重用于其他内容。但是，在实际销毁对象之前，即使没有强引用，弱引用也一直能返回该对象
        self.weak = weakref.WeakValueDictionary()
        #deque 类似 list，实现两端append和pop
        #如果 maxlen 没有指定或者是 None ，deques 可以增长到任意长度。否则，deque就限定到指定最大长度。一旦限定长度的deque满了，当新项加入时，同样数量的项就从另一端弹出
        self.strong = collections.deque(maxlen=MAX_len)
    # 声明静态方法,可以调用Dict.nowTime()
    @staticmethod
    def nowTime():
        return int(time.time())

    def settle(self, key, value):
        # strongRef作为强引用避免被回收
        self.weak[key] = strongRef = LocalCache.Dict(value)
        # 放入定大队列，弹出元素马上被回收
        self.strong.append(strongRef)

    def retrieve(self, key):
        #dict.get(key, default=None) default表示键值不存在时的返回值
        #self.weak是一个WeakValueDictionary
        value = self.weak.get(key, self.notFound)
        if value is not self.notFound:
            #value的两个属性是result和expire
            expire = value[r'expire']
            
            if self.nowTime() > expire:
                return self.notFound
            else:
                return value
        else:
            return self.notFound



def funcCache(expire=0):
    caches = LocalCache(100)

    def BIG_wrap(func):
        # wraps 装饰器， 
        @wraps(func)
        def SMALL_wrap(*args, **kwargs):
            
            key = str(func) + str(args) + str(kwargs)
            
            result = caches.retrieve(key)
            if result is LocalCache.notFound:
                result = func(*args, **kwargs)
                #这里的expire是3600
                caches.settle(key, {r'result': result, r'expire': expire + caches.nowTime()})
                result = caches.retrieve(key)
            return result

        return SMALL_wrap

    return BIG_wrap

EXP=3600
#这样 test_cache(v) 等价于 funcCache(test_cache(v))
@funcCache(expire=EXP)
def test_cache(v):
    # 模拟任务处理时常1秒
    
    time.sleep(1)
    
    return v



# 获取aiml的安装路径 
def get_module_dir(name):
    path = getattr(sys.modules[name], '__file__', None)
    if not path:
        raise AttributeError('module %s has not attribute __file__' % name)
    return os.path.dirname(os.path.abspath(path))
 
# 补充路径名称 

alice_path = get_module_dir('aiml') + '/botdata/alice'

# 切换到语料库所在工作目录 
os.chdir(alice_path) 

# 创建机器人alice对象 
alice = aiml.Kernel() 


# 这里做一个判断
# 如果是第一次加载语料库，就进入else部分，读取数据，同时保存资料至bot_brain.brn
# 如果是之后再加载语料库，就不需要读取所有数据了，直接读取保存数据bot_brain.brn
if os.path.isfile("bot_brain.brn"):
    alice.bootstrap(brainFile = "bot_brain.brn")
else:
    alice.learn("startup.xml") 
    alice.respond('LOAD ALICE') 
    alice.saveBrain("bot_brain.brn")


# 正式开始聊天 
coll=defaultdict(list)
while True:
    message = input("Enter your message >> ")    
    if ("exit" == message):# 如果输入exit，程序退出
        break
        response = alice.respond(message) # 机器人应答
        
        print(response)
        break # 结束循环
    response = alice.respond(message) # 机器人应答
    coll[message].append(response)
    print(response)
coll
