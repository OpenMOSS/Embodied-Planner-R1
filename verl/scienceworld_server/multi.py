import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
import uuid
import logging

from scienceworld import ScienceWorldEnv

logger = logging.getLogger(__name__)

class MultiScienceWorldEnv:
    """
    ScienceWorldEnv的多线程管理器
    在多个线程上创建和运行独立的环境实例，确保返回值与原始环境一致
    """
    
    def __init__(self, envStepLimit: int = 100, batch_size: int = 4):
        """
        初始化多线程环境管理器
        
        Args:
            envStepLimit: 环境步数限制，默认为100
            batch_size: 要创建的环境数量，也是线程池大小
        """
        self.envStepLimit = envStepLimit
        self.batch_size = batch_size
        
        # 创建线程池，用于并行操作
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        
        # 环境实例字典
        self.envs = {}
        self.envs_id = [] # 确保返回的 ids 顺序一致
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 在初始化时创建所有环境
        self._create_all_envs()
    
    def _create_all_envs(self):
        """创建所有环境实例"""
        # 并行创建环境
        futures = []
        env_ids = []
        
        for _ in range(self.batch_size):
            env_id = str(uuid.uuid4())
            env_ids.append(env_id)
            future = self.executor.submit(self._create_env, env_id)
            futures.append((env_id, future))
        
        # 等待所有环境创建完成
        for env_id, future in futures:
            try:
                future.result()  # 获取结果，检查是否有异常
                logger.info(f"环境 {env_id} 创建成功")
            except Exception as e:
                logger.error(f"创建环境 {env_id} 失败: {e}")
    
    def _create_env(self, env_id: str):
        """创建单个环境实例"""
        # 创建ScienceWorldEnv实例
        env = ScienceWorldEnv(envStepLimit=self.envStepLimit)
        
        # 注册环境
        with self.lock:
            self.envs[env_id] = env
            self.envs_id.append(env_id)
    
    def get_env_ids(self) -> List[str]:
        """获取所有可用环境ID"""
        with self.lock:
            return self.envs_id.copy()
    
    def _load(self, env_id: str, taskName: str, variationIdx: int = 0, 
             simplificationStr: str = "", generateGoldPath: bool = False) -> None:
        """
        Args:
            env_id: 环境ID
            taskName: 任务名称
            variationIdx: 变体索引，默认为0
            simplificationStr: 简化字符串，默认为空字符串
            generateGoldPath: 是否生成黄金路径，默认为False
            
        Returns:
            None (与原始ScienceWorldEnv.load保持一致)
        """
        if env_id not in self.envs:
            raise ValueError(f"环境 {env_id} 不存在")
        
        # 获取环境实例
        env = self.envs[env_id]
        
        # 加载任务
        env.load(taskName, variationIdx, simplificationStr, generateGoldPath)
    
    def load(self, task: List[str], var: List[int], 
                 simplificationStr: str = "", generateGoldPath: bool = False):
        """
        为所有环境加载任务
        
        Args:
            taskName: 任务名称
            variationIdx: 变体索引，默认为0
            simplificationStr: 简化字符串，默认为空字符串
            generateGoldPath: 是否生成黄金路径，默认为False
        """
        futures = []
        env_ids = self.get_env_ids()
        
        # 并行加载任务
        for i, env_id in enumerate(env_ids):
            future = self.executor.submit(
                self._load, env_id, task[i], var[i], simplificationStr, generateGoldPath
            )
            futures.append(future)
        
        # 等待所有加载完成
        concurrent.futures.wait(futures)
    
    def _reset(self, env_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        重置指定环境
        
        Args:
            env_id: 环境ID
            
        Returns:
            Tuple[str, Dict[str, Any]]: 与原始ScienceWorldEnv.reset返回值相同
                (observation, info)
        """
        if env_id not in self.envs:
            raise ValueError(f"环境 {env_id} 不存在")
        
        # 获取环境实例
        env = self.envs[env_id]
        
        # 重置环境
        initialObs, initialDict = env.reset()
        
        return initialObs
    
    def reset(self) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        """
        重置所有环境
        
        Returns:
            Dict[str, Tuple[str, Dict[str, Any]]]: 环境ID到(observation, info)的映射
        """
        futures = {}
        env_ids = self.get_env_ids()
        
        # 并行重置环境
        for env_id in env_ids:
            future = self.executor.submit(self._reset, env_id)
            futures[env_id] = future
        
        # 收集结果
        results = {}
        for env_id, future in futures.items():
            try:
                results[env_id] = future.result()
            except Exception as e:
                logger.error(f"重置环境 {env_id} 失败: {e}")
                results[env_id] = None
        obs = []
        for env_id in env_ids:
            obs.append(results[env_id])
        return obs
    
    def _step(self, env_id: str, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        在指定环境中执行步骤
        
        Args:
            env_id: 环境ID
            action: 动作字符串
            
        Returns:
            Tuple[str, float, bool, Dict[str, Any]]: 与原始ScienceWorldEnv.step返回值相同
                (observation, reward, done, info)
        """
        if env_id not in self.envs:
            raise ValueError(f"环境 {env_id} 不存在")
        
        # 获取环境实例
        env = self.envs[env_id]
        
        # 执行步骤
        observation, score, isCompleted, info = env.step(action)
        
        return observation, score, isCompleted, info
    
    def step(self, actions: List[str]) -> Dict[str, Tuple[str, float, bool, Dict[str, Any]]]:
        """
        在所有环境上执行步骤
        
        Args:
            actions: 动作列表，将依次应用于环境列表
                
        Returns:
            Dict[str, Tuple[str, float, bool, Dict[str, Any]]]: 环境ID到(observation, reward, done, info)的映射
        """
        futures = {}
        env_ids = self.get_env_ids()
        
        # 并行执行步骤
        for i, env_id in enumerate(env_ids):
            future = self.executor.submit(self._step, env_id, actions[i])
            futures[env_id] = future
        
        # 收集结果
        results = {}
        for env_id, future in futures.items():
            try:
                results[env_id] = future.result()
            except Exception as e:
                logger.error(f"执行环境 {env_id} 的步骤失败: {e}")
                # 返回一个空结果
                results[env_id] = (None, 0.0, True, {})
        
        step_res = []
        for env_id in env_ids:
            step_res.append(results[env_id])
        return tuple(zip(*step_res))
    
    def _taskdescription(self, env_id: str):
        if env_id not in self.envs:
            raise ValueError(f"环境 {env_id} 不存在")
        
        # 获取环境实例
        env = self.envs[env_id]

        return str(env.taskdescription())

    def get_task_description(self):
        futures = {}
        env_ids = self.get_env_ids()
        
        # 并行执行步骤
        for env_id in env_ids:
            future = self.executor.submit(self._taskdescription, env_id)
            futures[env_id] = future
        
        # 收集结果
        results = {}
        for env_id, future in futures.items():
            try:
                results[env_id] = future.result()
            except Exception as e:
                logger.error(f"执行环境 {env_id} 的步骤失败: {e}")
        task = []
        for env_id in env_ids:
            task.append(results[env_id])
        return task

    def get_task_names(self, env_id: str = None) -> List[str]:
        """
        获取可用任务名称列表
        
        Args:
            env_id: 环境ID，默认为None（使用第一个可用环境）
            
        Returns:
            List[str]: 任务名称列表
        """
        if env_id is None:
            env_id = self.get_env_ids()[0]
        
        if env_id not in self.envs:
            raise ValueError(f"环境 {env_id} 不存在")
        
        return self.envs[env_id].get_task_names()
    
    def close(self):
        """关闭所有环境和资源"""
        # 关闭所有环境
        with self.lock:
            for env_id, env in list(self.envs.items()):
                try:
                    if hasattr(env, 'close'):
                        env.close()
                except Exception as e:
                    logger.error(f"关闭环境 {env_id} 失败: {e}")
        
        # 清空环境字典
        self.envs.clear()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """析构函数，确保资源释放"""
        try:
            self.close()
        except:
            pass

if __name__ == "__main__":
    # 创建多环境管理器，自动创建8个环境实例
    multi_env = MultiScienceWorldEnv(envStepLimit=2, batch_size=2)

    try:
        # 获取所有环境ID
        env_ids = multi_env.get_env_ids()
        print(f"创建了 {len(env_ids)} 个环境实例")
        
        # 获取可用任务名称列表
        task_names = multi_env.get_task_names()
        print(f"可用任务名称: {task_names}")
        
        # 为所有环境加载相同的任务
        task_name = task_names[0]  # 使用第一个可用任务
        data = [{'task': task_name, 'var': 0}, {'task': task_name, 'var': 1}]
        task = [task_name, task_name]
        var = [0,1]
        # print(data)
        multi_env.load(task, var)
        
        # 重置所有环境
        reset_results = multi_env.reset()
        print(reset_results)
        task = multi_env.get_task_description()
        # print(task)
        # 输出初始观察
        # for env_id, (observation, info) in reset_results.items():
        #     print(f"环境 {env_id} 初始观察: {observation[:50]}...")
        # print(multi_env.get_env_ids())
        # print(multi_env.taskdescription_all())
        # 运行多步交互
        for i in range(5):
            # 准备每个环境的动作
            actions = {}
            # for env_id in env_ids:
            #     # 这里可以基于环境状态生成不同的动作
            #     actions[env_id] = "look around"  # 替换为实际动作
            actions = ['look around'] * 2
            # 并行执行所有环境的步骤
            results = multi_env.step(actions)
            
            obs, reward, done, info = results
            # print(obs)
            print(i)
            print(reward)
            print(done)
            
            # 处理结果
            # for env_id, result in results.items():
                # observation, reward, done, info = result
                # print(f"环境 {env_id} 步骤 {step}: obs {observation}, 奖励={reward}, 完成={done}")

    finally:
        # 关闭所有资源
        multi_env.close()