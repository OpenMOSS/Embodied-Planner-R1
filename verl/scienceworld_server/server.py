
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import sys
import time
import logging
import argparse
from datetime import datetime

from multi import MultiScienceWorldEnv

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="SCIWorld 服务器")
parser.add_argument("--port", type=int, default=8000, help="服务器端口")
parser.add_argument("--server_id", type=str, default="", help="服务器ID，用于日志区分")
args = parser.parse_args()

# 根据服务器ID生成日志文件名
server_id = args.server_id if args.server_id else f"server_{args.port}"
log_filename = f"ScienceWorld_{server_id}.log"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(f"ScienceWorld_server_{server_id}")

app = FastAPI()

class TrajRequest(BaseModel):
    task: List[str]
    var: List[int]

class ActionRequest(BaseModel):
    actions: List[str]

class Response(BaseModel):
    observations: List[str]
    tasks: Optional[List[str]] = None
    dones: List[bool]
    scores: List[float]  
    # infos: List[Dict[str, Any]]

# Global environment variable
env = None

@app.post("/reset", response_model=Response)
async def reset(request: TrajRequest) -> Response:
    global env
    start_time = time.time()
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    logger.info(f"[{request_id}] 收到重置请求: task={request.task}, var={request.var},batch_size={len(request.task)}")
    
    try:
        task = request.task
        var = request.var
        # batch_size = request.batch_size

        # 检查是否存在旧环境，如果存在则关闭
        if env is not None:
            logger.info(f"[{request_id}] 正在关闭旧环境...")
            try:
                # 尝试调用环境的close方法（如果存在）
                if hasattr(env, 'close'):
                    env.close()
                    logger.info(f"[{request_id}] 旧环境已成功关闭")
                else:
                    logger.warning(f"[{request_id}] 旧环境没有明确的关闭方法，将直接替换")
            except Exception as close_error:
                logger.warning(f"[{request_id}] 关闭旧环境时出现异常: {str(close_error)}")
                # 即使关闭失败，我们仍然会继续创建新环境
            
            del env
        
        logger.info(f"[{request_id}] 正在初始化环境...")
        # Initialize the environment
        env = MultiScienceWorldEnv(envStepLimit=1000 , batch_size=len(task))  

        env.load(task=task, var=var)
        
        logger.info(f"[{request_id}] 环境初始化完成，正在重置环境...")
        # Reset the environment and get initial observations

        obs = env.reset()
        tasks = env.get_task_description()
        logger.info(f"[{request_id}] 环境重置完成，获得初始观察值数量: {len(obs)}")

        # Prepare the response
        response = Response(
            observations=obs,
            tasks=tasks,
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
            # infos=infos
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
        "environment_status": global_env_status,
        "server_id": server_id,
        "port": args.port
    }


@app.on_event("startup")
async def startup_event():
    logger.info(f"=== SciWorld 服务 {server_id} 启动 ===")
    logger.info(f"启动时间: {datetime.now().isoformat()}")
    logger.info(f"服务运行在端口: {args.port}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"=== SciWorld 服务 {server_id} 关闭 ===")
    logger.info(f"关闭时间: {datetime.now().isoformat()}")

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    logger.info(f"正在启动 SciWorld 服务 {server_id} 在端口 {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)