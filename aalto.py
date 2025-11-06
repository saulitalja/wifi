#!/usr/bin/env python3
import sys
import time
import threading
import subprocess
import platform
import re
from collections import defaultdict, deque
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ------------------ Helpers (kanava <-> taajuus) ------------------
def channel_to_freq_mhz(channel):
    try:
        ch = int(channel)
    except Exception:
        return None
    if 1 <= ch <= 14:
        return 2407 + 5 * ch
    table = {
        36: 5180, 40: 5200, 44: 5220, 48: 5240,
        52: 5260, 56:5280, 60:5300, 64:5320,
        100:5500,104:5520,108:5540,112:5560,116:5580,120:5600,124:5620,128:5640,
        132:5660,136:5680,140:5700,144:5720,
        149:5745,153:5765,157:5785,161:5805,165:5825
    }
    return table.get(ch)

# ------------------ Platform parsers (kevyet, kuten aiemmin) ------------------
def parse_nmcli(output):
    networks = []
    lines = output.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        # Split by two or more spaces (nmcli aligns in columns)
        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
        if not parts:
            continue
        # If header line, skip
        if parts[0].lower().startswith('ssid') or parts[0].lower().startswith('in-use'):
            continue
        ssid = parts[0]
        chan = None
        sig = None
        # try find channel (a small number) and signal (percent)
        for token in parts[1:]:
            if re.match(r'^\d+$', token):
                chan = token
            m = re.search(r'(\d+)$', token)
            if m and (token.endswith('%') or (int(m.group(1)) <= 100 and 'signal' in ' '.join(parts).lower())):
                sig = float(m.group(1))
        freq = channel_to_freq_mhz(int(chan)) if chan else None
        dbm = None
        if sig is not None:
            # approximate conversion % -> dBm (very rough) (0% -> -100, 100% -> -50)
            dbm = (sig / 100.0) * 50.0 - 100.0
        networks.append({'ssid': ssid or '<hidden>', 'freq_mhz': freq, 'dbm': dbm})
    return networks

def parse_iwlist(output):
    networks = []
    cells = re.split(r'Cell \d+ - ', output)
    for c in cells:
        if not c.strip():
            continue
        ssid_m = re.search(r'ESSID:"([^"]*)"', c)
        freq_m = re.search(r'Frequency:([0-9.]+)\s*GHz', c)
        ch_m = re.search(r'Channel[:=]?\s*([0-9]+)', c)
        ssid = ssid_m.group(1) if ssid_m else '<hidden>'
        freq = None
        if freq_m:
            freq = float(freq_m.group(1)) * 1000.0
        elif ch_m:
            freq = channel_to_freq_mhz(int(ch_m.group(1)))
        dbm = None
        dbm_m = re.search(r'Signal level[=\:]\s*([-\d]+)\s*dBm', c)
        if dbm_m:
            dbm = float(dbm_m.group(1))
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

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
                dbm = (pct/100.0)*50.0 - 100.0
        freq = channel_to_freq_mhz(channel) if channel else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

def parse_airport(output):
    networks = []
    lines = output.splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r'\s{2,}', line.strip())
        ssid = parts[0] if parts else '<hidden>'
        # try to find channel token like "36" or "11"
        ch = None
        rssi = None
        for token in parts:
            if re.match(r'^\d{1,3}(?:,.*)?$', token):
                ch = int(token.split(',')[0])
            if re.match(r'^-?\d+\s*$', token):
                val = int(token.strip())
                if -120 < val < 0:
                    rssi = val
        freq = channel_to_freq_mhz(ch) if ch else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': rssi})
    return networks

# ------------------ Platform scan functions ------------------
def scan_linux():
    try:
        out = subprocess.check_output(['nmcli', '-f', 'SSID,CHAN,SIGNAL', 'device', 'wifi', 'list'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6)
        parsed = parse_nmcli(out)
        if parsed:
            return parsed
    except Exception:
        pass
    try:
        iwconfig_out = subprocess.check_output(['iwconfig'], stderr=subprocess.DEVNULL, universal_newlines=True, timeout=4)
        m = re.search(r'([a-zA-Z0-9]+)\s+IEEE 802.11', iwconfig_out)
        iface = m.group(1) if m else 'wlan0'
        out = subprocess.check_output(['sudo', 'iwlist', iface, 'scan'], stderr=subprocess.DEVNULL, universal_newlines=True, timeout=12)
        return parse_iwlist(out)
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
    try:
        out = subprocess.check_output(['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6)
        return parse_airport(out)
    except Exception:
        try:
            out = subprocess.check_output(['/usr/sbin/airport', '-s'], stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6)
            return parse_airport(out)
        except Exception:
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

# ------------------ Live plotting data structures ------------------
MAX_POINTS = 60  # how many time-steps to keep (e.g. 60 samples)
UPDATE_INTERVAL_MS = 2000  # update every 2000 ms (2s)

# map ssid -> deque of last dBm values
history = defaultdict(lambda: deque(maxlen=MAX_POINTS))
# maintain fixed time axis (seconds relative)
time_axis = deque(maxlen=MAX_POINTS)

# colors
_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

_lock = threading.Lock()
_last_scan_time = None

def scanner_thread_func(stop_event):
    global _last_scan_time
    while not stop_event.is_set():
        nets = scan_wifi()
        t = time.time()
        with _lock:
            # push current time
            if not time_axis or (t - (time_axis[-1] if time_axis else 0) >= 0.9):
                time_axis.append(t)
            # update histories
            seen = set()
            for net in nets:
                ssid = net.get('ssid') or '<hidden>'
                dbm = net.get('dbm')
                if dbm is None:
                    # if no dBm, give a default weak value (so curve exists)
                    dbm = -100.0
                history[ssid].append(dbm)
                seen.add(ssid)
            # for SSIDs not seen this round, append NaN to advance their timeline
            for s in list(history.keys()):
                if s not in seen:
                    # append a NaN or low value; NaN will create gap in plot
                    history[s].append(np.nan)
            _last_scan_time = t
        # sleep until next scan
        time.sleep(max(0.5, UPDATE_INTERVAL_MS/1000.0 - 0.1))

# ------------------ Plotting ------------------
fig, ax = plt.subplots(figsize=(12,6))
lines = {}
fills = {}
legend_texts = []

def init_plot():
    ax.clear()
    ax.set_title("Aktiivinen Wi-Fi -aaltokuvaaja (signaali dBm vs aika)")
    ax.set_xlabel("Aika (s, viimeiset ~{})".format(MAX_POINTS))
    ax.set_ylabel("Signal strength (dBm)")
    ax.set_ylim(-110, -30)
    ax.grid(True, alpha=0.3)

def update_plot(frame):
    with _lock:
        if not time_axis:
            return
        times = np.array(time_axis)
        t0 = times[0]
        times_rel = times - times[-1]  # show negative (past) relative seconds
        ssids = sorted(history.keys(), key=lambda s: np.nanmax(np.array(history[s])) if history[s] else -999)
        # keep top N networks to avoid clutter
        top_ssids = ssids[:12]
        ax.clear()
        init_plot()
        # plot each ssid as filled waveform
        for i, ssid in enumerate(top_ssids):
            vals = np.array(history[ssid])
            if vals.size == 0:
                continue
            color = _color_cycle[i % len(_color_cycle)]
            # smoother: interpolate missing NaNs for plotting; keep NaNs to show gaps
            x = times_rel
            y = vals
            # simpler smoothing: replace single NaNs with -120 to push curve down
            y_plot = np.where(np.isnan(y), -120.0, y)
            # vertical offset or stacking? we overlay with alpha
            ax.plot(x, y_plot, label=ssid, color=color, linewidth=1.5)
            ax.fill_between(x, y_plot, -120, where=~np.isnan(y), alpha=0.12, color=color)
        ax.set_xlim(times_rel[0], 0.5)
        ax.legend(loc='upper left', fontsize='small', ncol=1)
        ax.set_ylim(-120, -30)
        ax.set_xlabel("Aika (s, viimeiset ~{})".format(MAX_POINTS))
        ax.set_ylabel("Signal strength (dBm)")
        ax.set_title("Aktiivinen Wi-Fi -aaltokuvaaja (päivittyy joka {} ms)".format(UPDATE_INTERVAL_MS))
        ax.grid(True, alpha=0.25)

# ------------------ Main ------------------
def main():
    stop_event = threading.Event()
    scanner = threading.Thread(target=scanner_thread_func, args=(stop_event,), daemon=True)
    scanner.start()
    init_plot()
    ani = animation.FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        scanner.join(timeout=2)

if __name__ == '__main__':
    main()
