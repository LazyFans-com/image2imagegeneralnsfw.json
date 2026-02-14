"""
Custom RunPod handler for Qwen Rapid AIO SFW image-to-image workflow.

Accepts pre-signed URLs for input/output instead of base64 blobs.
Builds the ComfyUI workflow internally from simple parameters.
"""

import json
import random
import time
import urllib.parse
import urllib.request
import uuid

import requests
import runpod
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_API_URL = f"http://{COMFY_HOST}"

DEFAULT_NEGATIVE_PROMPT = (
    "paint, blurry, extra limbs, low resolution, cartoon, anime, "
    "plastic skin, grainy, noisy, watermarks, text, bad anatomy, "
    "unnatural pose, artifacts"
)


# ---------------------------------------------------------------------------
# ComfyUI server helpers
# ---------------------------------------------------------------------------

def check_server(timeout: int = 600):
    """Poll ComfyUI HTTP server until it responds (or timeout)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFY_API_URL}/system_stats", timeout=5)
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(f"ComfyUI server not ready after {timeout}s")


def queue_workflow(workflow: dict, client_id: str) -> str:
    """POST workflow to ComfyUI /prompt and return the prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{COMFY_API_URL}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ComfyUI prompt error: {data['error']}")
    return data["prompt_id"]


def get_history(prompt_id: str) -> dict:
    """GET /history/{prompt_id}."""
    resp = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json()


def get_image_data(filename: str, subfolder: str, folder_type: str) -> bytes:
    """GET /view to fetch output image bytes."""
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    resp = requests.get(f"{COMFY_API_URL}/view?{params}")
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# WebSocket monitoring with reconnect
# ---------------------------------------------------------------------------

def _create_ws(client_id: str) -> websocket.WebSocket:
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")
    return ws


def _attempt_websocket_reconnect(client_id: str, max_retries: int = 3) -> websocket.WebSocket:
    for attempt in range(max_retries):
        try:
            return _create_ws(client_id)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    raise ConnectionError("Failed to reconnect to ComfyUI websocket")


def wait_for_completion(prompt_id: str, client_id: str, timeout: int = 300):
    """Monitor websocket until the queued prompt completes or errors."""
    ws = _create_ws(client_id)
    start = time.time()

    try:
        while time.time() - start < timeout:
            try:
                ws.settimeout(10)
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except (
                websocket.WebSocketConnectionClosedException,
                ConnectionError,
                BrokenPipeError,
            ):
                ws = _attempt_websocket_reconnect(client_id)
                continue

            if isinstance(raw, bytes):
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "executing" and data.get("prompt_id") == prompt_id:
                if data.get("node") is None:
                    return  # execution finished
            elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                raise RuntimeError(
                    f"ComfyUI execution error on node {data.get('node_type')}: "
                    f"{data.get('exception_message', 'unknown')}"
                )
    finally:
        try:
            ws.close()
        except Exception:
            pass

    raise TimeoutError(f"Workflow execution timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Image transfer helpers
# ---------------------------------------------------------------------------

def download_image(url: str) -> bytes:
    """Download image from a pre-signed GET URL."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def upload_image_to_comfy(image_bytes: bytes, filename: str):
    """Upload image to ComfyUI /upload/image endpoint."""
    resp = requests.post(
        f"{COMFY_API_URL}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json()


def upload_to_s3(image_bytes: bytes, put_url: str):
    """Upload image bytes to a pre-signed PUT URL."""
    resp = requests.put(
        put_url,
        data=image_bytes,
        headers={"Content-Type": "image/png"},
        timeout=120,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 6,
    cfg: float = 1,
    denoise: float = 1,
) -> dict:
    """Build the 12-node Qwen Rapid AIO SFW workflow dict."""
    return {
        "1": {
            "inputs": {"strength": 1, "model": ["4", 0]},
            "class_type": "CFGNorm",
            "_meta": {"title": "CFGNorm"},
        },
        "3": {
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "EmptyLatentImage"},
        },
        "4": {
            "inputs": {"shift": 3.1, "model": ["10", 0]},
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        "5": {
            "inputs": {
                "prompt": negative_prompt,
                "clip": ["9", 0],
                "vae": ["10", 2],
                "image1": ["11", 0],
            },
            "class_type": "TextEncodeQwenImageEditPlus",
            "_meta": {"title": "TextEncodeQwenImageEditPlus"},
        },
        "6": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "beta",
                "denoise": denoise,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "7": {
            "inputs": {
                "prompt": prompt,
                "clip": ["9", 0],
                "vae": ["10", 2],
                "image1": ["11", 0],
            },
            "class_type": "TextEncodeQwenImageEditPlus",
            "_meta": {"title": "TextEncodeQwenImageEditPlus"},
        },
        "8": {
            "inputs": {"samples": ["6", 0], "vae": ["10", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAEDecode"},
        },
        "9": {
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "CLIPLoader"},
        },
        "10": {
            "inputs": {"ckpt_name": "Qwen-Rapid-AIO-SFW-v23.safetensors"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "CheckpointLoaderSimple"},
        },
        "11": {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Body"},
        },
        "12": {
            "inputs": {
                "filename_prefix": "output_",
                "images": ["8", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "SaveImage"},
        },
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """RunPod serverless handler entry point."""
    inp = event.get("input", {})

    # --- Validate required fields ---
    input_image_url = inp.get("input_image_url")
    output_upload_url = inp.get("output_upload_url")
    prompt = inp.get("prompt")

    missing = []
    if not input_image_url:
        missing.append("input_image_url")
    if not output_upload_url:
        missing.append("output_upload_url")
    if not prompt:
        missing.append("prompt")
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}

    # --- Optional parameters with defaults ---
    negative_prompt = inp.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    width = inp.get("width", 1080)
    height = inp.get("height", 1920)
    seed = inp.get("seed", random.randint(0, 2**53))

    # --- Step 1: Wait for ComfyUI ---
    try:
        check_server()
    except TimeoutError as e:
        return {"error": str(e)}

    # --- Step 2: Download input image ---
    try:
        image_bytes = download_image(input_image_url)
    except Exception as e:
        return {"error": f"Failed to download input image: {e}"}

    # --- Step 3: Upload image to ComfyUI ---
    comfy_filename = f"input_{uuid.uuid4().hex[:8]}.png"
    try:
        upload_image_to_comfy(image_bytes, comfy_filename)
    except Exception as e:
        return {"error": f"Failed to upload image to ComfyUI: {e}"}

    # --- Step 4: Build & queue workflow ---
    workflow = build_workflow(
        image_name=comfy_filename,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
    )

    client_id = uuid.uuid4().hex
    try:
        prompt_id = queue_workflow(workflow, client_id)
    except Exception as e:
        return {"error": f"Failed to queue workflow: {e}"}

    # --- Step 5: Wait for execution ---
    try:
        wait_for_completion(prompt_id, client_id)
    except Exception as e:
        return {"error": f"Workflow execution failed: {e}"}

    # --- Step 6: Fetch output image ---
    try:
        history = get_history(prompt_id)
        outputs = history[prompt_id]["outputs"]
        # Node 12 is SaveImage
        images = outputs["12"]["images"]
        output_image = images[0]
        output_bytes = get_image_data(
            output_image["filename"],
            output_image.get("subfolder", ""),
            output_image.get("type", "output"),
        )
    except Exception as e:
        return {"error": f"Failed to fetch output image: {e}"}

    # --- Step 7: Upload to S3 ---
    try:
        upload_to_s3(output_bytes, output_upload_url)
    except Exception as e:
        return {"error": f"Failed to upload output to S3: {e}"}

    # Return clean URL (strip query params which contain the signature)
    clean_url = output_upload_url.split("?")[0]
    return {"status": "success", "output_url": clean_url}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
