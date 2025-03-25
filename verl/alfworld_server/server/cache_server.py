from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import sys
import time
import logging
from datetime import datetime

sys.path.append("/inspire/ssd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zyfei/open-embodied-r1")

from alfworld_server.alfworld_server_lite.tw_env import get_tw_env
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("alfworld_server.log")
    ]
)
logger = logging.getLogger("alfworld_server")

app = FastAPI()

class TrajRequest(BaseModel):
    game_file: str
    batch_size: int

class ActionRequest(BaseModel):
    actions: List[str]

class Response(BaseModel):
    observations: List[str]
    dones: List[bool]
    scores: List[float]  
    # infos: List[Dict[str, Any]]

# Global environment variable
env = None

# 添加在全局变量区域
env_cache = {}

# def get_cached_tw_env(game_file, batch_size, max_nb_steps_per_episode=100):
#     """获取缓存的TextWorld环境，如果不存在则创建"""
#     cache_key = f"{game_file}_{batch_size}_{max_nb_steps_per_episode}"
    
#     if cache_key in env_cache:
#         logger.info(f"使用缓存环境: {cache_key}")
#         return env_cache[cache_key]
    
#     # 创建新环境
#     logger.info(f"创建新环境: {cache_key}")
#     env = get_tw_env(game_file, batch_size, max_nb_steps_per_episode)
#     env_cache[cache_key] = env
#     return env

# 添加在全局变量区域
MAX_CACHE_SIZE = 1000  # 最多缓存10个不同的环境

def get_cached_tw_env(game_file, batch_size, max_nb_steps_per_episode=100):
    """获取缓存的TextWorld环境，如果不存在则创建"""
    cache_key = f"{game_file}_{batch_size}_{max_nb_steps_per_episode}"
    
    if cache_key in env_cache:
        logger.info(f"使用缓存环境: {cache_key}")
        return env_cache[cache_key]
    
    # 如果缓存已满，删除最早创建的环境
    if len(env_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(env_cache))
        try:
            if hasattr(env_cache[oldest_key], 'close'):
                env_cache[oldest_key].close()
        except Exception as e:
            logger.warning(f"关闭最早环境 {oldest_key} 时出现错误: {str(e)}")
        
        del env_cache[oldest_key]
        logger.info(f"缓存已满，删除最早环境: {oldest_key}")
    
    # 创建新环境
    logger.info(f"创建新环境: {cache_key}")
    env = get_tw_env(game_file, batch_size, max_nb_steps_per_episode)
    env_cache[cache_key] = env
    return env

@app.post("/reset", response_model=Response)
async def reset(request: TrajRequest) -> Response:
    global env
    start_time = time.time()
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    logger.info(f"[{request_id}] 收到重置请求: game_file={request.game_file}, batch_size={request.batch_size}")
    
    try:
        game_file = request.game_file
        batch_size = request.batch_size
        
        # 使用缓存机制获取环境
        env = get_cached_tw_env(game_file, batch_size)
        
        logger.info(f"[{request_id}] 正在重置环境...")
        # Reset the environment and get initial observations
        obs, infos = env.reset()
        
        logger.info(f"[{request_id}] 环境重置完成，获得初始观察值数量: {len(obs)}")
        
        # Prepare the response
        response = Response(
            observations=obs,
            dones=[False] * len(obs),  # Initial state is not done
            scores=[0.0] * len(obs),   # Initial score is 0
            # infos=infos
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"[{request_id}] 重置请求处理完成，耗时: {elapsed_time:.4f}秒")
        return response
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"[{request_id}] 重置请求处理失败，耗时: {elapsed_time:.4f}秒，错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

class CacheRequest(BaseModel):
    game_file: str
    batch_size: int

@app.post("/preload")
async def preload_environment(request: CacheRequest):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[{request_id}] 收到环境预加载请求: game_file={request.game_file}, batch_size={request.batch_size}")
    
    try:
        # 预加载环境到缓存
        _ = get_cached_tw_env(request.game_file, request.batch_size)
        return {"status": "success", "message": "Environment preloaded successfully"}
    except Exception as e:
        logger.error(f"[{request_id}] 预加载环境失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/clear_cache")
async def clear_environment_cache():
    global env_cache
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[{request_id}] 收到清理环境缓存请求")
    
    try:
        # 关闭所有缓存的环境
        for key, cached_env in env_cache.items():
            try:
                if hasattr(cached_env, 'close'):
                    cached_env.close()
            except Exception as e:
                logger.warning(f"[{request_id}] 关闭环境 {key} 时出现错误: {str(e)}")
        
        # 清空缓存
        cache_size = len(env_cache)
        env_cache = {}
        
        return {"status": "success", "message": f"Cleared {cache_size} environments from cache"}
    except Exception as e:
        logger.error(f"[{request_id}] 清理环境缓存失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=Response)
async def step(request: ActionRequest) -> Response:
    global env
    start_time = time.time()
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    logger.info(f"[{request_id}] 收到步进请求: actions数量={len(request.actions)}")
    
    if env is None:
        logger.error(f"[{request_id}] 环境未初始化错误")
        raise HTTPException(status_code=400, detail="Environment not initialized, call /reset before /step!")
    
    try:
        logger.info(f"[{request_id}] 执行动作中...")
        # Execute the actions in the environment
        obs, scores, dones, infos = env.step(request.actions)
        
        logger.info(f"[{request_id}] 动作执行完成，获得观察值数量: {len(obs)}")
        logger.info(f"[{request_id}] 已执行动作: {' '.join([str(d) + ' |' for d in request.actions])}")
        logger.info(f"[{request_id}] 观察值: {' '.join([str(d) + ' |' for d in obs])}")
        logger.info(f"[{request_id}] 已完成状态: {' '.join([str(d) for d in dones])}")
        
        # Return the results
        response = Response(
            observations=obs,
            dones=dones,
            scores=scores,
            infos=infos
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"[{request_id}] 步进请求处理完成，耗时: {elapsed_time:.4f}秒")
        return response
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"[{request_id}] 步进请求处理失败，耗时: {elapsed_time:.4f}秒，错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[{request_id}] 收到健康检查请求")
    
    global_env_status = "initialized" if env is not None else "not initialized"
    logger.info(f"[{request_id}] 当前环境状态: {global_env_status}")
    
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "environment_status": global_env_status
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=== ALFWorld 服务启动 ===")
    logger.info(f"启动时间: {datetime.now().isoformat()}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== ALFWorld 服务关闭 ===")
    logger.info(f"关闭时间: {datetime.now().isoformat()}")

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    logger.info("正在启动 ALFWorld 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)