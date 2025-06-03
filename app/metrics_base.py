import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from confluent_kafka import Producer
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logger = logging.getLogger(__name__)

class BaseMetricsProducer:
    """
    Base class for service-specific metrics producers.
    Each service should create its own producer instance with its service name.
    """
    def __init__(
        self,
        service_name: str,
        bootstrap_servers: str = None,
        topic: str = "metrics",
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_workers: int = 4
    ):
        self.service_name = service_name
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = topic
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Thread pool for Kafka operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize Kafka producer
        self.producer = Producer({
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': f'{service_name}-metrics',
            'queue.buffering.max.messages': self.batch_size,
            'queue.buffering.max.ms': int(self.flush_interval * 1000)
        })
        
        # Thread-safe batch management
        self.batch = []
        self._batch_lock = asyncio.Lock()
    
    def _delivery_report(self, err, msg):
        """Callback for message delivery reports"""
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    async def _flush_batch(self):
        """Flush the current batch to Kafka asynchronously"""
        async with self._batch_lock:
            if not self.batch:
                return
                
            current_batch = self.batch.copy()
            self.batch = []
        
        # Run Kafka operations in thread pool with timeout
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._produce_batch,
                    current_batch
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error("Batch processing timed out")
            # Requeue failed batch items
            async with self._batch_lock:
                self.batch.extend(current_batch)
    
    def _produce_batch(self, batch):
        """Produce a batch of messages to Kafka (runs in thread pool)"""
        for message in batch:
            try:
                self.producer.produce(
                    self.topic,
                    json.dumps(message).encode('utf-8'),
                    callback=self._delivery_report
                )
            except Exception as e:
                logger.error(f"Failed to produce message: {str(e)}")
        
        self.producer.flush()
    
    async def _periodic_flush(self):
        """Periodically flush the batch"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                # Ensure we flush one last time before shutting down
                await self._flush_batch()
                raise
            except Exception as e:
                logger.error(f"Error in periodic flush: {str(e)}")
                continue
    
    async def send_metric(
        self,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Send a metric to Kafka asynchronously"""
        message = {
            "type": "metric",
            "timestamp": datetime.now(datetime.UTC).timestamp(),
            "service_name": self.service_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metadata": metadata,
            "tags": tags
        }
        
        try:
            async with self._batch_lock:
                self.batch.append(message)
                
                # Flush if batch is full
                if len(self.batch) >= self.batch_size:
                    asyncio.create_task(self._flush_batch())
        except Exception as e:
            logger.error(f"Error sending metric: {str(e)}")
            # Attempt to send metric directly if batching fails
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._produce_batch,
                    [message]
                )
            except Exception as e:
                logger.error(f"Failed to send metric directly: {str(e)}")
    
    async def send_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Send an event to Kafka asynchronously"""
        message = {
            "type": "event",
            "timestamp": datetime.utcnow().timestamp(),
            "service_name": self.service_name,
            "event_type": event_type,
            "event_data": event_data
        }
        
        try:
            async with self._batch_lock:
                self.batch.append(message)
                
                # Flush if batch is full
                if len(self.batch) >= self.batch_size:
                    asyncio.create_task(self._flush_batch())
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            # Attempt to send event directly if batching fails
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._produce_batch,
                    [message]
                )
            except Exception as e:
                logger.error(f"Failed to send event directly: {str(e)}")
    
    async def close(self):
        """Clean up resources asynchronously"""
        try:
            # Cancel the periodic flush task
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            
            # Flush any remaining metrics with timeout
            try:
                await asyncio.wait_for(self._flush_batch(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout while flushing final metrics")
            
            # Close the thread pool
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Ensure thread pool is closed even if there's an error
            self.executor.shutdown(wait=True)
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        try:
            # Run cleanup in event loop if it exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except:
            pass 