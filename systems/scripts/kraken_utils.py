import requests
import time
import hashlib
import hmac
import base64
from urllib.parse import urlencode

KRAKEN_API_URL = "https://api.kraken.com"

def _kraken_request(endpoint: str, data: dict, api_key: str, api_secret: str) -> dict:
    url_path = f"/0/private/{endpoint}"
    url = KRAKEN_API_URL + url_path

    nonce = str(int(1000 * time.time()))
    data["nonce"] = nonce

    post_data = urlencode(data)
    encoded = (nonce + post_data).encode()
    message = url_path.encode() + hashlib.sha256(encoded).digest()

    signature = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
    sig_digest = base64.b64encode(signature.digest())

    headers = {
        "API-Key": api_key,
        "API-Sign": sig_digest.decode()
    }

    resp = requests.post(url, headers=headers, data=data)
    result = resp.json()

    if "error" in result and result["error"]:
        raise Exception(f"Kraken API error: {result['error']}")

    return result
