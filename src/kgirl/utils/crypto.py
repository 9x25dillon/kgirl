from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

def verify_signature(message: bytes, signature: bytes, pubkey_hex: str) -> bool:
    try:
        vk = VerifyKey(bytes.fromhex(pubkey_hex))
        vk.verify(message, signature)
        return True
    except BadSignatureError:
        return False