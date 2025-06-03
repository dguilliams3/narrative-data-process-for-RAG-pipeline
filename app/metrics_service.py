import sqlite3
import json
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web
import os

# Initialize logging
LOG_FILE_PATH = "logs/metrics_service.log"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Metrics Service")

# Database setup
DB_PATH = "metrics.db"

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create metrics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            service_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metadata TEXT,
            tags TEXT
        )
    ''')
    
    # Create events table for event-driven metrics
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            service_name TEXT NOT NULL,
            event_data TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    conn.commit()
    conn.close()

class Metric(BaseModel):
    service_name: str
    metric_name: str
    metric_value: float
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None

class Event(BaseModel):
    event_type: str
    service_name: str
    event_data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    logger.info("Metrics service started and database initialized")

@app.post("/metrics")
async def record_metric(metric: Metric):
    """Record a new metric"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO metrics (service_name, metric_name, metric_value, metadata, tags)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            metric.service_name,
            metric.metric_name,
            metric.metric_value,
            json.dumps(metric.metadata) if metric.metadata else None,
            json.dumps(metric.tags) if metric.tags else None
        ))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Metric recorded"}
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def record_event(event: Event):
    """Record a new event"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO events (event_type, service_name, event_data)
            VALUES (?, ?, ?)
        ''', (
            event.event_type,
            event.service_name,
            json.dumps(event.event_data)
        ))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Event recorded"}
    except Exception as e:
        logger.error(f"Error recording event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{service_name}")
async def get_metrics(service_name: str, metric_name: Optional[str] = None, limit: int = 100):
    """Get metrics for a specific service"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        query = '''
            SELECT timestamp, metric_name, metric_value, metadata, tags
            FROM metrics
            WHERE service_name = ?
        '''
        params = [service_name]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        results = c.fetchall()
        
        metrics = []
        for row in results:
            metrics.append({
                "timestamp": row[0],
                "metric_name": row[1],
                "metric_value": row[2],
                "metadata": json.loads(row[3]) if row[3] else None,
                "tags": json.loads(row[4]) if row[4] else None
            })
        
        conn.close()
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/{service_name}")
async def get_events(service_name: str, event_type: Optional[str] = None, limit: int = 100):
    """Get events for a specific service"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        query = '''
            SELECT timestamp, event_type, event_data, status
            FROM events
            WHERE service_name = ?
        '''
        params = [service_name]
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        results = c.fetchall()
        
        events = []
        for row in results:
            events.append({
                "timestamp": row[0],
                "event_type": row[1],
                "event_data": json.loads(row[2]),
                "status": row[3]
            })
        
        conn.close()
        return events
    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 