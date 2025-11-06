#!/usr/bin/env python3
import sys
import time
import threading
import subprocess
import platform
import re
from collections import defaultdict, deque
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# macOS CoreWLAN -tuki
if platform.system().lower() == "darwin":
    try:
        import objc
        from CoreWLAN import CWInterface
    except Exception:
        CWInterface = None

# ------------------ Helpers ------------------
def channel_to_freq_mhz(channel):
    try:
        ch = int(channel)
    except Exception:
        return None
    if 1 <= ch <= 14:
        return 2407 + 5 * ch
    table = {
        36: 5180, 40: 5200, 44: 5220, 48: 5240,
        52: 5260, 56: 5280, 60: 5300, 64: 5320,
        100: 5500, 104: 5520, 108: 5540, 112: 5560, 116: 5580,
        120: 5600, 124: 5620, 128: 5640, 132: 5660, 136: 5680,
        140: 5700, 144: 5720, 149: 5745, 153: 5765,
        157: 5785, 161: 5805, 165: 5825
    }
    return table.get(ch)

# ------------------ Parsers ------------------
def parse_netsh(output):
    networks = []
    ssid_blocks = re.split(r'\r?\n\s*SSID\s+\d+\s*:\s*', output)
    for blk in ssid_blocks[1:]:
        lines = blk.splitlines()
        if not lines:
            continue
        ssid = lines[0].strip() or '<hidden>'
        channel = None
        dbm = None
        for L in lines[1:]:
            mchan = re.search(r'Channel\s*:\s*(\d+)', L)
            if mchan:
                channel = int(mchan.group(1))
            msignal = re.search(r'Signal\s*:\s*([0-9]+)%', L)
            if msignal:
                pct = int(msignal.group(1))
                dbm = (pct / 100.0) * 50.0 - 100.0
        freq = channel_to_freq_mhz(channel) if channel else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

def parse_nmcli(output):
    networks = []
    lines = output.strip().splitlines()
    for line in lines:
        if not line.strip() or line.startswith("SSID"):
            continue
        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
        if not parts:
            continue
        ssid = parts[0]
        chan = None
        sig = None
        for token in parts[1:]:
            if re.match(r'^\d+$', token):
                chan = token
            m = re.search(r'(\d+)$', token)
            if m and token.endswith('%'):
                sig = float(m.group(1))
        freq = channel_to_freq_mhz(int(chan)) if chan else None
        dbm = (sig / 100.0) * 50.0 - 100.0 if sig is not None else None
        networks.append({'ssid': ssid or '<hidden>', 'freq_mhz': freq, 'dbm': dbm})
    return networks

# ------------------ Platform scan functions ------------------
def scan_linux():
    try:
        out = subprocess.check_output(['nmcli', '-f', 'SSID,CHAN,SIGNAL', 'device', 'wifi', 'list'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6)
        return parse_nmcli(out)
    except Exception:
        return []

def scan_windows():
    try:
        out = subprocess.check_output(['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True, timeout=8)
        return parse_netsh(out)
    except Exception:
        return []

def scan_macos():
    if CWInterface:
        iface = CWInterface.interface()
        if iface is None:
            return []
        nets = iface.scanForNetworksWithName_error_(None, None)[0]
        networks = []
        for n in nets:
            ssid = n.ssid() or "<hidden>"
            rssi = n.rssiValue()
            freq = n.wlanChannel().channelNumber()
            freq_mhz = channel_to_freq_mhz(freq)
            snr = n.noiseMeasurement()  # jos saatavilla
            networks.append({'ssid': ssid, 'freq_mhz': freq_mhz, 'dbm': rssi, 'snr': snr})
        return networks
    return []

def scan_wifi():
    osn = platform.system().lower()
    if 'linux' in osn:
        return scan_linux()
    elif 'windows' in osn:
        return scan_windows()
    elif 'darwin' in osn:
        return scan_macos()
    else:
        return []

# ------------------ Live plotting ------------------
MAX_POINTS = 60
UPDATE_INTERVAL_MS = 2000
history = defaultdict(lambda: deque(maxlen=MAX_POINTS))
time_axis = deque(maxlen=MAX_POINTS)
_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
_lock = threading.Lock()

def scanner_thread_func(stop_event):
    while not stop_event.is_set():
        nets = scan_wifi()
        t = time.time()
        with _lock:
            if not time_axis or (t - (time_axis[-1] if time_axis else 0) >= 0.9):
                time_axis.append(t)
            seen = set()
            for net in nets:
                ssid = net.get('ssid') or '<hidden>'
                dbm = net.get('dbm', -100.0)
                history[ssid].append(dbm)
                seen.add(ssid)
            for s in list(history.keys()):
                if s not in seen:
                    history[s].append(np.nan)
        time.sleep(max(0.5, UPDATE_INTERVAL_MS / 1000.0 - 0.1))

fig, ax = plt.subplots(figsize=(12,6))

def init_plot():
    ax.set_title("Wi-Fi signaalitasot (dBm)")
    ax.set_xlabel("Aika (s, viimeiset ~{})".format(MAX_POINTS))
    ax.set_ylabel("Signaali (dBm)")
    ax.set_ylim(-120, -30)
    ax.grid(True, alpha=0.3)

def update_plot(frame):
    with _lock:
        if not time_axis:
            return
        times = np.array(time_axis)
        times_rel = times - times[-1]
        ssids = sorted(history.keys(), key=lambda s: np.nanmax(np.array(history[s])) if history[s] else -999)
        top_ssids = ssids[:10]
        ax.clear()
        init_plot()
        for i, ssid in enumerate(top_ssids):
            vals = np.array(history[ssid])
            color = _color_cycle[i % len(_color_cycle)]
            y = np.where(np.isnan(vals), -120.0, vals)
            ax.plot(times_rel, y, label=ssid, color=color, linewidth=1.5)
            ax.fill_between(times_rel, y, -120, alpha=0.15, color=color)
        ax.legend(loc='upper left', fontsize='small')
        ax.set_xlim(times_rel[0], 0.5)

def main():
    stop_event = threading.Event()
    scanner = threading.Thread(target=scanner_thread_func, args=(stop_event,), daemon=True)
    scanner.start()
    init_plot()
    ani = animation.FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS)
    try:
        plt.show()
    finally:
        stop_event.set()
        scanner.join(timeout=2)

if __name__ == '__main__':
    main()
