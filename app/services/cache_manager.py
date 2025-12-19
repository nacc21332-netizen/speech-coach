"""
Двухуровневый кэш-менеджер: память + диск.
L1 (память): 0.1ms - горячие данные
L2 (диск): 10-50ms - холодные данные
"""
import asyncio
import logging
from typing import Any, Optional
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class TwoLevelCache:
    """
    Двухуровневый кэш для оптимальной производительности.
    
    L1: In-memory TTL кэш (быстрый, для горячих данных)
    L2: Диск кэш (медленный, но персистентный)
    """
    
    def __init__(
        self,
        disk_cache=None,
        memory_maxsize: int = 100,
        ttl_seconds: int = 3600
    ):
        """
        Args:
            disk_cache: AnalysisCache instance (диск кэш)
            memory_maxsize: Максимальное количество записей в памяти
            ttl_seconds: Время жизни кэша в секундах
        """
        # L1: In-memory кэш для горячих данных
        self.memory = TTLCache(maxsize=memory_maxsize, ttl=ttl_seconds)
        
        # L2: Диск кэш для холодных данных
        self.disk = disk_cache
        
        # Статистика для мониторинга
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0
        self.ttl_seconds = ttl_seconds
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Получить значение с приоритетом L1 → L2.
        
        Args:
            key: Ключ кэша
            
        Returns:
            Значение из кэша или None если не найдено
        """
        
        # L1: Память (0.1ms)
        if key in self.memory:
            self.hits_l1 += 1
            logger.debug(f"Cache L1 hit: {key}")
            return self.memory[key]
        
        # L2: Диск (10-50ms)
        if self.disk:
            try:
                value = await asyncio.to_thread(self.disk.get_by_key, key)
                if value is not None:
                    # Переместить в L1 для следующего раза
                    self.memory[key] = value
                    self.hits_l2 += 1
                    logger.debug(f"Cache L2 hit (promoted to L1): {key}")
                    return value
            except Exception as e:
                logger.debug(f"L2 cache error: {e}")
        
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """
        Сохранить значение в оба уровня кэша.
        
        Args:
            key: Ключ кэша
            value: Значение для сохранения
        """
        # Сохраняем в L1 (память)
        self.memory[key] = value
        
        # Асинхронно сохраняем в L2 (диск)
        if self.disk:
            try:
                await asyncio.to_thread(self.disk.set_by_key, key, value)
                logger.debug(f"Cache set (L1+L2): {key}")
            except Exception as e:
                logger.warning(f"Failed to set disk cache: {e}")
                logger.info(f"Cache set (L1 only): {key}")
    
    def stats(self) -> dict:
        """
        Получить статистику эффективности кэша.
        
        Returns:
            dict с метриками кэша (hits, misses, hit_rate, etc.)
        """
        total = self.hits_l1 + self.hits_l2 + self.misses
        
        if total > 0:
            hit_rate = (self.hits_l1 + self.hits_l2) / total * 100
            l1_percentage = self.hits_l1 / total * 100
            l2_percentage = self.hits_l2 / total * 100
        else:
            hit_rate = 0
            l1_percentage = 0
            l2_percentage = 0
        
        return {
            "l1_memory_hits": self.hits_l1,
            "l2_disk_hits": self.hits_l2,
            "misses": self.misses,
            "total_requests": total,
            "overall_hit_rate": f"{hit_rate:.1f}%",
            "l1_hit_percentage": f"{l1_percentage:.1f}%",
            "l2_hit_percentage": f"{l2_percentage:.1f}%",
            "memory_size": len(self.memory),
            "memory_maxsize": self.memory.maxsize,
            "ttl_seconds": self.ttl_seconds,
        }
    
    async def clear(self) -> None:
        """Очистить оба уровня кэша"""
        self.memory.clear()
        
        if self.disk:
            try:
                await asyncio.to_thread(self.disk.clear_old)
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
        
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0
        logger.info("Cache cleared")
