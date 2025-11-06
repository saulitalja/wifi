#!/usr/bin/env python3
import time
import threading
import subprocess
import platform
import re
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Kanava -> taajuus (MHz) ---
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

# --- Parsinta ---
def parse_nmcli(output):
    networks = []
    lines = output.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
        if not parts or parts[0].lower().startswith('ssid'):
            continue
        ssid = parts[0]
        chan = None
        sig = None
        for token in parts[1:]:
            if re.match(r'^\d+$', token):
                chan = token
            m = re.search(r'(\d+)$', token)
            if m and (token.endswith('%') or int(m.group(1)) <= 100):
                sig = float(m.group(1))
        freq = channel_to_freq_mhz(int(chan)) if chan else None
        dbm = (sig / 100.0) * 50.0 - 100.0 if sig is not None else None
        networks.append({'ssid': ssid or '<hidden>', 'freq_mhz': freq, 'dbm': dbm})
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
                dbm = (pct / 100.0) * 50.0 - 100.0
        freq = channel_to_freq_mhz(channel) if channel else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

# --- Skanneri ---
def scan_wifi():
    osn = platform.system().lower()
    try:
        if 'linux' in osn:
            out = subprocess.check_output(
                ['nmcli', '-f', 'SSID,CHAN,SIGNAL', 'device', 'wifi', 'list'],
                stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6)
            return parse_nmcli(out)
        elif 'windows' in osn:
            out = subprocess.check_output(
                ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                stderr=subprocess.DEVNULL, universal_newlines=True, timeout=8)
            return parse_netsh(out)
    except Exception:
        return []
    return []

# --- Mittausdata ---
MAX_POINTS = 60
UPDATE_INTERVAL_MS = 2000

history_rssi = defaultdict(lambda: deque(maxlen=MAX_POINTS))
history_snr = defaultdict(lambda: deque(maxlen=MAX_POINTS))
history_srri = defaultdict(lambda: deque(maxlen=MAX_POINTS))
history_freq = defaultdict(lambda: deque(maxlen=MAX_POINTS))
time_axis = deque(maxlen=MAX_POINTS)
_lock = threading.Lock()

def dbm_to_snr(dbm):
    return max(0.0, dbm + 100.0) if dbm is not None else 0.0

def snr_to_srri(snr):
    return max(0.0, min(100.0, (snr / 70.0) * 100.0))

def scanner_thread(stop_event):
    while not stop_event.is_set():
        nets = scan_wifi()
        t = time.time()
        with _lock:
            if not time_axis or (t - (time_axis[-1] if time_axis else 0) >= 1.0):
                time_axis.append(t)
            seen = set()
            for net in nets:
                ssid = net.get('ssid') or '<hidden>'
                dbm = net.get('dbm', -100.0)
                freq = net.get('freq_mhz', np.nan)
                snr = dbm_to_snr(dbm)
                srri = snr_to_srri(snr)
                history_rssi[ssid].append(dbm)
                history_snr[ssid].append(snr)
                history_srri[ssid].append(srri)
                history_freq[ssid].append(freq)
                seen.add(ssid)
            for s in list(history_rssi.keys()):
                if s not in seen:
                    history_rssi[s].append(np.nan)
                    history_snr[s].append(np.nan)
                    history_srri[s].append(np.nan)
                    history_freq[s].append(np.nan)
        time.sleep(UPDATE_INTERVAL_MS / 1000.0)

# --- Piirto ---
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
plt.subplots_adjust(hspace=0.4)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def update_plot(frame):
    with _lock:
        if not time_axis:
            return
        times = np.array(time_axis)
        rel_time = times - times[-1]
        for ax in axes:
            ax.clear()
            ax.grid(True, alpha=0.3)

        titles = [
            "RSSI (Received Signal Strength, dBm)",
            "SNR (Signal-to-Noise Ratio, dB)",
            "SRRI (Scaled SNR, 0–100)",
            "Taajuus (MHz)"
        ]
        ylabels = ["dBm", "dB", "SRRI (0–100)", "MHz"]
        for ax, t, yl in zip(axes, titles, ylabels):
            ax.set_title(t)
            ax.set_ylabel(yl)
        axes[-1].set_xlabel("Aika (s, viimeiset ~60)")

        ssids = sorted(history_rssi.keys(), key=lambda s: np.nanmax(np.array(history_rssi[s])) if history_rssi[s] else -999)
        top_ssids = ssids[:8]

        for i, ssid in enumerate(top_ssids):
            c = colors[i % len(colors)]
            rssi = np.array(history_rssi[ssid])
            snr = np.array(history_snr[ssid])
            srri = np.array(history_srri[ssid])
            freq = np.array(history_freq[ssid])
            axes[0].plot(rel_time, rssi, color=c, label=ssid)
            axes[1].plot(rel_time, snr, color=c, label=ssid)
            axes[2].plot(rel_time, srri, color=c, label=ssid)
            axes[3].plot(rel_time, freq, color=c, label=ssid)

        axes[0].set_ylim(-110, -30)
        axes[1].set_ylim(0, 70)
        axes[2].set_ylim(0, 100)
        axes[3].set_ylim(2400, 5900)

        for ax in axes:
            ax.legend(loc="upper left", fontsize="small")

def main():
    stop_event = threading.Event()
    thread = threading.Thread(target=scanner_thread, args=(stop_event,), daemon=True)
    thread.start()
    ani = animation.FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2)

if __name__ == "__main__":
    main()
