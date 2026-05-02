#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from multibios.fictrac_runtime import prepare_fictrac_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe FicTrac subprocess launch modes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fictrac-bin", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("direct", "cmd", "powershell", "console_handles"),
        default="direct",
        help="Launch FicTrac directly or through cmd.exe",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_dirs = prepare_fictrac_runtime()
    if runtime_dirs:
        print("runtime_dirs")
        for runtime_dir in runtime_dirs:
            print(runtime_dir)

    env = os.environ.copy()
    fictrac_bin = args.fictrac_bin.resolve()
    config = args.config.resolve()

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(("", 5556))
    udp_socket.settimeout(args.timeout)

    popen_kwargs: dict[str, object] = {
        "cwd": str(fictrac_bin.parent),
        "env": env,
    }

    console_in = None
    console_out = None
    if args.mode == "direct":
        command = [str(fictrac_bin), str(config)]
    elif args.mode == "cmd":
        command = ["cmd", "/d", "/c", subprocess.list2cmdline([str(fictrac_bin), str(config)])]
    elif args.mode == "powershell":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"& '{fictrac_bin}' '{config}'",
        ]
    else:
        command = [str(fictrac_bin), str(config)]
        console_in = open("CONIN$", "r")
        console_out = open("CONOUT$", "w")
        popen_kwargs["stdin"] = console_in
        popen_kwargs["stdout"] = console_out
        popen_kwargs["stderr"] = console_out
        popen_kwargs["close_fds"] = False

    process = subprocess.Popen(command, **popen_kwargs)
    print(f"pid {process.pid}")
    try:
        payload, address = udp_socket.recvfrom(4096)
        print(f"got_packet {len(payload)} {address}")
        return 0
    except TimeoutError:
        print(f"timeout returncode={process.poll()}")
        return 1
    finally:
        process.terminate()
        udp_socket.close()
        if console_in is not None:
            console_in.close()
        if console_out is not None:
            console_out.close()


if __name__ == "__main__":
    raise SystemExit(main())