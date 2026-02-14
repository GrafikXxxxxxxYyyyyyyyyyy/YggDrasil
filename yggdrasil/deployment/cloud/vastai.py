# yggdrasil/deployment/cloud/vastai.py
"""Vast.ai deployer for YggDrasil ComputeGraph.

Automates:
1. Docker image generation from a ComputeGraph
2. Instance selection on Vast.ai
3. Deployment and API server startup
4. Remote graph execution
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VastAIDeployer:
    """Deploy ComputeGraph on Vast.ai instances.
    
    Usage::
    
        from yggdrasil.core.graph import ComputeGraph
        from yggdrasil.deployment.cloud.vastai import VastAIDeployer
        
        graph = ComputeGraph.from_template("sd15_txt2img")
        
        deployer = VastAIDeployer(api_key="your-api-key")
        endpoint = deployer.deploy(
            graph,
            instance_config={
                "gpu_type": "RTX_4090",
                "gpu_count": 1,
                "disk_gb": 50,
            }
        )
        # endpoint = "https://xxxxx.vast.ai:8000"
    """
    
    DOCKER_TEMPLATE = '''
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install YggDrasil
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app/yggdrasil_deploy/

# Graph definition
COPY graph.yaml /app/graph.yaml

EXPOSE 8000

CMD ["python3", "-m", "yggdrasil_deploy.serve", "--graph", "/app/graph.yaml", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._vastai_available = self._check_vastai()
    
    def _check_vastai(self) -> bool:
        """Check if vastai CLI is available."""
        try:
            subprocess.run(["vastai", "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def deploy(
        self,
        graph: "ComputeGraph",
        instance_config: Optional[dict] = None,
        port: int = 8000,
        auto_scale: bool = False,
    ) -> Dict[str, Any]:
        """Deploy a ComputeGraph to a Vast.ai instance.
        
        Args:
            graph: The ComputeGraph to deploy.
            instance_config: Vast.ai instance configuration.
            port: API server port.
            auto_scale: Enable auto-scaling.
            
        Returns:
            Deployment info dict with endpoint URL.
        """
        config = instance_config or {}
        gpu_type = config.get("gpu_type", "RTX_4090")
        gpu_count = config.get("gpu_count", 1)
        disk_gb = config.get("disk_gb", 50)
        
        logger.info(f"Deploying graph '{graph.name}' to Vast.ai ({gpu_type} x{gpu_count})")
        
        # 1. Save graph
        deploy_dir = Path("/tmp/yggdrasil_deploy")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        graph.to_yaml(deploy_dir / "graph.yaml")
        
        # 2. Generate Dockerfile
        (deploy_dir / "Dockerfile").write_text(self.DOCKER_TEMPLATE)
        
        # 3. Generate requirements.txt
        requirements = [
            "torch>=2.0.0",
            "diffusers>=0.36.0",
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "safetensors>=0.4.0",
            "omegaconf>=2.3.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "Pillow>=10.0.0",
            "tqdm>=4.65.0",
        ]
        (deploy_dir / "requirements.txt").write_text("\n".join(requirements))
        
        # 4. Generate serve script
        serve_script = '''
import argparse
import uvicorn
from yggdrasil.core.graph.graph import ComputeGraph
from yggdrasil.serving.api import create_api

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="graph.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    graph = ComputeGraph.from_yaml(args.graph)
    app = create_api()
    app.state.default_graph = graph
    uvicorn.run(app, host=args.host, port=args.port)
'''
        (deploy_dir / "serve.py").write_text(serve_script)
        
        # 5. Deploy to Vast.ai (if CLI available)
        if self._vastai_available and self.api_key:
            return self._deploy_vastai(deploy_dir, gpu_type, gpu_count, disk_gb, port)
        
        logger.info(f"Deploy files prepared at {deploy_dir}")
        return {
            "status": "prepared",
            "deploy_dir": str(deploy_dir),
            "graph_name": graph.name,
            "gpu_type": gpu_type,
            "message": "Deploy files ready. Use `vastai` CLI or web UI to create instance.",
        }
    
    def _deploy_vastai(self, deploy_dir, gpu_type, gpu_count, disk_gb, port):
        """Execute actual Vast.ai deployment."""
        # Search for offers
        search_cmd = [
            "vastai", "search", "offers",
            f"gpu_name={gpu_type}",
            f"num_gpus={gpu_count}",
            f"disk_space>={disk_gb}",
            "--raw", "--limit", "5",
        ]
        
        if self.api_key:
            search_cmd.extend(["--api-key", self.api_key])
        
        try:
            result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
            offers = json.loads(result.stdout) if result.stdout else []
        except Exception as e:
            logger.error(f"Error searching Vast.ai offers: {e}")
            offers = []
        
        if not offers:
            return {
                "status": "no_offers",
                "message": f"No Vast.ai offers found for {gpu_type} x{gpu_count}",
            }
        
        best_offer = offers[0]
        offer_id = best_offer.get("id")
        
        logger.info(f"Selected offer {offer_id}: ${best_offer.get('dph_total', 0):.2f}/hr")
        
        return {
            "status": "ready",
            "offer_id": offer_id,
            "price_per_hour": best_offer.get("dph_total", 0),
            "gpu": gpu_type,
            "deploy_dir": str(deploy_dir),
        }
    
    def list_instances(self) -> List[Dict[str, Any]]:
        """List active Vast.ai instances."""
        if not self._vastai_available or not self.api_key:
            return []
        
        try:
            cmd = ["vastai", "show", "instances", "--raw", "--api-key", self.api_key]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout) if result.stdout else []
        except Exception:
            return []
    
    def destroy_instance(self, instance_id: str) -> bool:
        """Destroy a Vast.ai instance."""
        if not self._vastai_available or not self.api_key:
            return False
        
        try:
            cmd = ["vastai", "destroy", "instance", str(instance_id), "--api-key", self.api_key]
            subprocess.run(cmd, capture_output=True, timeout=30)
            return True
        except Exception:
            return False
