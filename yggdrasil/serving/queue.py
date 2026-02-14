"""Job queue for concurrent generation requests."""
from __future__ import annotations

import asyncio
import uuid
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A generation job in the queue."""
    id: str
    model_id: str
    params: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    
    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


class JobQueue:
    """In-memory job queue for managing generation requests.
    
    For production, replace with Redis-backed queue.
    """
    
    def __init__(self, max_concurrent: int = 1, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self._jobs: Dict[str, Job] = {}
        self._pending: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running: int = 0
        self._lock = asyncio.Lock()
    
    async def submit(self, model_id: str, params: Dict[str, Any]) -> Job:
        """Submit a new job to the queue.
        
        Returns:
            The created Job
        """
        job = Job(
            id=str(uuid.uuid4()),
            model_id=model_id,
            params=params,
        )
        self._jobs[job.id] = job
        await self._pending.put(job)
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> list:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        return False
    
    async def process_jobs(self, handler: Callable):
        """Process jobs from the queue.
        
        Args:
            handler: Async function that takes a Job and produces results
        """
        while True:
            job = await self._pending.get()
            
            if job.status == JobStatus.CANCELLED:
                continue
            
            async with self._lock:
                self._running += 1
            
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            
            try:
                result = await handler(job)
                job.result = result
                job.status = JobStatus.COMPLETED
            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = time.time()
                async with self._lock:
                    self._running -= 1
    
    def cleanup(self, max_age_seconds: float = 3600):
        """Remove old completed/failed jobs."""
        now = time.time()
        to_remove = [
            jid for jid, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            and (now - job.created_at) > max_age_seconds
        ]
        for jid in to_remove:
            del self._jobs[jid]
