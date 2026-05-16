#!/usr/bin/env python3
"""
Run rave preprocess with all parallelism disabled.
Patches multiprocessing.Pool and Manager to inline equivalents BEFORE
importing rave, so the Manager server thread is never started.

Usage (same args as rave preprocess):
    python3 cloud/preprocess_single.py \
        --input_path data/ \
        --output_path processed/rave_preprocessed \
        --sampling_rate 44100 \
        --num_signal 262144
"""
import multiprocessing
import multiprocessing.pool
import concurrent.futures
import queue
import threading


class _InlinePool:
    """Drop-in for multiprocessing.Pool — runs all work in the main thread."""
    def __init__(self, processes=None, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def imap(self, func, iterable, chunksize=1):
        return map(func, iterable)
    def imap_unordered(self, func, iterable, chunksize=1):
        return map(func, iterable)
    def map(self, func, iterable, chunksize=None):
        return list(map(func, iterable))
    def starmap(self, func, iterable, chunksize=None):
        return [func(*a) for a in iterable]
    def apply(self, func, args=(), kwargs={}):
        return func(*args, **kwargs)
    def apply_async(self, func, args=(), kwargs={}, callback=None, error_callback=None):
        try:
            r = func(*args, **kwargs)
            if callback: callback(r)
        except Exception as e:
            if error_callback: error_callback(e)
    def close(self): pass
    def join(self): pass
    def terminate(self): pass


class _InlineManager:
    """Drop-in for multiprocessing.Manager() — uses thread-safe in-process objects."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def start(self): pass
    def shutdown(self): pass
    def Queue(self, maxsize=0): return queue.Queue(maxsize)
    def JoinableQueue(self, maxsize=0): return queue.Queue(maxsize)
    def Event(self): return threading.Event()
    def Lock(self): return threading.Lock()
    def RLock(self): return threading.RLock()
    def Semaphore(self, value=1): return threading.Semaphore(value)
    def Value(self, typecode, value, *a):
        class _V:
            def __init__(self, v): self.value = v
        return _V(value)
    def list(self, *a): return list(*a) if a else []
    def dict(self, *a, **kw): return dict(*a, **kw)


class _InlineProcessPoolExecutor:
    def __init__(self, max_workers=None, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return list(map(fn, *iterables))
    def submit(self, fn, *a, **kw):
        class _F:
            def __init__(self, r): self._r = r
            def result(self, timeout=None): return self._r
            def exception(self, timeout=None): return None
        return _F(fn(*a, **kw))
    def shutdown(self, wait=True, cancel_futures=False): pass


# Patch before any rave import
multiprocessing.Pool = _InlinePool
multiprocessing.pool.Pool = _InlinePool
multiprocessing.Manager = lambda: _InlineManager()
concurrent.futures.ProcessPoolExecutor = _InlineProcessPoolExecutor

import sys
sys.argv = ['rave', 'preprocess'] + sys.argv[1:]

from rave.__main__ import cli
cli()
