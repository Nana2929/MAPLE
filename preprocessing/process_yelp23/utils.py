
import json
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result

    return wrapper

@timer
def read_jsonl(path: str):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def write_jsonl(lines: list, path: str):
    with open(path, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
@timer
def read_pickle(path: str):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path: str):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
