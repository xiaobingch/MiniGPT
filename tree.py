#!/usr/bin/env python3
import os
import sys


def show_tree(path, prefix='', depth=0, max_depth=3):
    if depth >= max_depth:
        return

    try:
        items = [
            name for name in os.listdir(path)
            if not name.startswith('.')
        ]
        items.sort(key=lambda name: (
            not os.path.isdir(os.path.join(path, name)),
            name.lower()
        ))
    except PermissionError:
        return

    for index, name in enumerate(items):
        full_path = os.path.join(path, name)
        is_last = index == len(items) - 1

        print(prefix + ('└── ' if is_last else '├── ') + name)

        if os.path.isdir(full_path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            show_tree(full_path, new_prefix, depth + 1, max_depth)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(os.path.abspath(path).rstrip('/') or '/')
    show_tree(path, max_depth=max_depth)