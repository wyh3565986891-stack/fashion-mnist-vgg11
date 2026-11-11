import gzip
import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

if __name__ == "__main__":
    raw = Path("data/FashionMNIST/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for filename, checksum in FILES.items():
        path = raw / filename.removesuffix(".gz")
        if path.exists():
            continue
        url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + filename
        with urlopen(url, timeout=45) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != checksum:
            raise ValueError("Unexpected checksum for " + filename)
        path.write_bytes(gzip.decompress(content))
        print(path, flush=True)
