
import cv2, numpy as np
from insightface import model_zoo

m = model_zoo.get_model('antelopev2')
print("TYPE:", type(m))
print("DIR (first 120):", [n for n in dir(m) if not n.startswith('_')][:120])
# show signature for a few likely methods if present
import inspect
for name in ("get", "get_embedding", "get_feat", "get_feats", "get_embeddings", "infer", "forward"):
    if hasattr(m, name):
        try:
            print(name, "->", inspect.signature(getattr(m, name)))
        except Exception:
            print(name, "-> signature unknown")
# quick trial on a bundled sample if available (optional)
try:
    import urllib.request
    url = "https://raw.githubusercontent.com/insightface/insightface/master/python/example/01.png"
    data = urllib.request.urlopen(url).read()
    import numpy as np, cv2, tempfile
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # try calling a few methods and print shapes / types
    for call in ("get", "get_embedding", "get_feat", "get_feats", "get_embeddings", "infer", "forward"):
        if hasattr(m, call):
            try:
                out = getattr(m, call)(img)
                print(call, "-> OK, type:", type(out), "len/shape:", getattr(out, '__len__', lambda: None)())
            except Exception as e:
                print(call, "-> ERROR:", e)
except Exception:
    pass

