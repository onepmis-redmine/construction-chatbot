import psutil
import asyncio
import httpx
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class SystemMonitor:
    def __init__(self):
        self.background_tasks = set()

    def collect_system_info(self):
        """시스템 리소스 사용량을 수집하여 반환합니다."""
        try:
            # 프로세스 정보 가져오기
            process = psutil.Process()
            
            # 메모리 정보
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # 시스템 전체 메모리 정보
            system_memory = psutil.virtual_memory()
            
            # CPU 사용량
            cpu_percent = process.cpu_percent(interval=1)
            
            # 디스크 사용량
            disk_usage = psutil.disk_usage('/')
            
            info = {
                "process_memory": {
                    "rss": f"{memory_info.rss / 1024 / 1024:.2f} MB",  # 실제 물리적 메모리 사용량
                    "vms": f"{memory_info.vms / 1024 / 1024:.2f} MB",  # 가상 메모리 사용량
                    "percent": f"{memory_percent:.2f}%"
                },
                "system_memory": {
                    "total": f"{system_memory.total / 1024 / 1024:.2f} MB",
                    "available": f"{system_memory.available / 1024 / 1024:.2f} MB",
                    "used": f"{system_memory.used / 1024 / 1024:.2f} MB",
                    "percent": f"{system_memory.percent}%"
                },
                "cpu": {
                    "percent": f"{cpu_percent}%"
                },
                "disk": {
                    "total": f"{disk_usage.total / 1024 / 1024:.2f} MB",
                    "used": f"{disk_usage.used / 1024 / 1024:.2f} MB",
                    "free": f"{disk_usage.free / 1024 / 1024:.2f} MB",
                    "percent": f"{disk_usage.percent}%"
                }
            }
            
            # 메모리 사용량이 80%를 초과하면 경고 로그
            if system_memory.percent > 80:
                logger.warning(f"메모리 사용량이 높습니다! ({system_memory.percent}%)")
            
            return info
        except Exception as e:
            logger.error(f"시스템 정보 수집 중 오류 발생: {e}")
            return {"error": str(e)}

    async def keep_alive(self):
        """서버가 슬립 상태로 전환되지 않도록 주기적으로 자체 핑을 수행하고 시스템 정보를 모니터링"""
        while True:
            try:
                # 5분마다 자체 ping (Render 무료 플랜 타임아웃은 일반적으로 15분)
                await asyncio.sleep(300)
                
                # 시스템 정보 수집 및 로깅
                system_info = self.collect_system_info()
                logger.info("시스템 리소스 사용량:")
                logger.info(f"- 프로세스 메모리 (RSS): {system_info['process_memory']['rss']}")
                logger.info(f"- 시스템 메모리 사용률: {system_info['system_memory']['percent']}")
                logger.info(f"- CPU 사용률: {system_info['cpu']['percent']}")
                
                # 메모리 사용량이 80%를 초과하면 경고 로그
                if float(system_info['system_memory']['percent'].rstrip('%')) > 80:
                    logger.warning(f"메모리 사용량이 높습니다! ({system_info['system_memory']['percent']})")
                
                async with httpx.AsyncClient() as client:
                    # 현재 서버 URL 동적 생성
                    host = "localhost"
                    port = "8000"
                    # 환경 변수 확인
                    if os.getenv("RENDER") == "true":
                        # Render 환경에서는 외부 URL 사용
                        response = await client.get("https://construction-chatbot-api.onrender.com/ping")
                    else:
                        # 개발 환경에서는 로컬 URL 사용
                        response = await client.get(f"http://{host}:{port}/ping")
                    
                    logger.info(f"자동 핑 응답: {response.status_code}")
                
            except Exception as e:
                logger.error(f"시스템 모니터링 중 오류 발생: {e}")
                continue

    def start_monitoring(self):
        """시스템 모니터링을 시작합니다."""
        keep_alive_task = asyncio.create_task(self.keep_alive())
        self.background_tasks.add(keep_alive_task)
        keep_alive_task.add_done_callback(self.background_tasks.discard)
        logger.info("시스템 모니터링 시작됨")

    async def stop_monitoring(self):
        """시스템 모니터링을 중지합니다."""
        for task in self.background_tasks:
            task.cancel()
        
        # 태스크가 완료될 때까지 대기
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("시스템 모니터링 중지됨") 