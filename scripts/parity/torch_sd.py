"""Read a PyTorch zip-format state_dict (.pth) with only the standard library + numpy."""
import pickle, zipfile, sys, json, numpy as np
DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_, "BFloat16Storage": None}
class _Stub:
    def __init__(self, *a, **k): pass
def load_state_dict(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    root = names[0].split("/")[0]
    storages = {}
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return name  # marker string
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, backward_hooks: data
            return _Stub
        def persistent_load(self, pid):
            typename, storage_type, key, location, numel = pid
            assert typename == "storage"
            dt = DTYPES[storage_type]
            if key not in storages:
                storages[key] = np.frombuffer(zf.read(f"{root}/data/{key}"), dtype=dt)
            return storages[key]
    def rebuild(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
        arr = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=tuple(s*storage.itemsize for s in stride))
        return np.array(arr)  # contiguous copy
    with zf.open(f"{root}/data.pkl") as f:
        return U(f).load()
if __name__ == "__main__":
    sd = load_state_dict(sys.argv[1])
    total = 0
    for k, v in sd.items():
        total += v.size
        print(f"{k:60s} {str(v.dtype):8s} {str(tuple(v.shape)):20s} {v.size:>10,d}")
    print("TOTAL params:", f"{total:,d}", "fp32 MB:", round(total*4/1e6,1), "fp16 MB:", round(total*2/1e6,1), "int8 MB(approx):", round(total/1e6,1))
