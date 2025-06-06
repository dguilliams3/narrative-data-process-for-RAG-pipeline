import json
import logging
import os
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsConsumer:
    """
    Consumer that reads metrics from Kafka and writes them to InfluxDB.
    """
    def __init__(
        self,
        kafka_bootstrap_servers: str = None,
        kafka_topic: str = "metrics",
        kafka_group_id: str = "metrics-consumer",
        influx_url: str = None,
        influx_token: str = None,
        influx_org: str = None,
        influx_bucket: str = None
    ):
        # Kafka configuration
        self.kafka_config = {
            'bootstrap.servers': kafka_bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            'group.id': kafka_group_id,
            'auto.offset.reset': 'earliest'
        }
        self.kafka_topic = kafka_topic
        
        # InfluxDB configuration
        self.influx_url = influx_url or os.getenv("INFLUX_URL", "http://localhost:8086")
        self.influx_token = influx_token or os.getenv("INFLUX_TOKEN")
        self.influx_org = influx_org or os.getenv("INFLUX_ORG", "my-org")
        self.influx_bucket = influx_bucket or os.getenv("INFLUX_BUCKET", "metrics")
        
        if not self.influx_token:
            raise ValueError("INFLUX_TOKEN environment variable is required")
        
        # Initialize Kafka consumer
        self.consumer = Consumer(self.kafka_config)
        self.consumer.subscribe([self.kafka_topic])
        
        # Initialize InfluxDB client
        self.influx_client = InfluxDBClient(
            url=self.influx_url,
            token=self.influx_token,
            org=self.influx_org
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
    def _write_metric_to_influx(self, metric_data):
        """Write a metric to InfluxDB"""
        try:
            # Create a point using metric_name and tag with service_name
            point = Point(metric_data["metric_name"])
            point = point.tag("service_name", metric_data.get("service_name", ""))
            
            # Add additional tags
            if metric_data.get("tags"):
                for key, value in metric_data["tags"].items():
                    point = point.tag(key, str(value))
            
            # Add metric field
            point = point.field("value", metric_data["metric_value"])
            
            # Add timestamp
            point = point.time(datetime.fromtimestamp(metric_data["timestamp"]))
            
            # Write to InfluxDB
            self.write_api.write(
                bucket=self.influx_bucket,
                record=point
            )
            
        except Exception as e:
            logger.error(f"Failed to write metric to InfluxDB: {str(e)}")
    
    def _write_event_to_influx(self, event_data):
        """Write an event to InfluxDB"""
        try:
            point = Point("events")
            
            # Add tags
            point = point.tag("event_type", event_data["event_type"])
            point = point.tag("service_name", event_data["service_name"])
            
            # Add fields from event_data
            for key, value in event_data["event_data"].items():
                if isinstance(value, (int, float)):
                    point = point.field(key, value)
                else:
                    point = point.field(key, str(value))
            
            # Add timestamp
            point = point.time(datetime.fromtimestamp(event_data["timestamp"]))
            
            # Write to InfluxDB
            self.write_api.write(
                bucket=self.influx_bucket,
                record=point
            )
            
        except Exception as e:
            logger.error(f"Failed to write event to InfluxDB: {str(e)}")
    
    def consume(self):
        """Start consuming messages from Kafka"""
        logger.info(f"Starting to consume messages from topic: {self.kafka_topic}")
        
        try:
            while True:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"Error while consuming message: {msg.error()}")
                    continue
                
                try:
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    if data["type"] == "metric":
                        self._write_metric_to_influx(data)
                    elif data["type"] == "event":
                        self._write_event_to_influx(data)
                    else:
                        logger.warning(f"Unknown message type: {data['type']}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.consumer.close()
            self.influx_client.close()
    
    def __del__(self):
        """Clean up resources"""
        try:
            self.consumer.close()
            self.influx_client.close()
        except:
            pass

if __name__ == "__main__":
    consumer = MetricsConsumer()
    consumer.consume() 