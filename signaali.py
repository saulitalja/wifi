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
        52: 5260, 56: 5280, 60: 5300, 64: 5320,
        100: 5500, 104: 5520, 108: 5540, 112: 5560, 116: 5580,
        120: 5600, 124: 5620, 128: 5640, 132: 5660,
        136: 5680, 140: 5700, 144: 5720,
        149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825
    }
    return table.get(ch)

# ------------------ Platform parsers ------------------
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
                dbm = (pct / 100.0) * 50.0 - 100.0  # approx −100…−50 dBm
        freq = channel_to_freq_mhz(channel) if channel else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

def scan_wifi():
    osn = platform.system().lower()
    try:
        if 'windows' in osn:
            out = subprocess.check_output(
                ['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                stderr=subprocess.DEVNULL, universal_newlines=True, timeout=8
            )
            return parse_netsh(out)
        elif 'linux' in osn:
            out = subprocess.check_output(
                ['nmcli', '-f', 'SSID,CHAN,SIGNAL', 'device', 'wifi', 'list'],
                stderr=subprocess.DEVNULL, universal_newlines=True, timeout=6
            )
            networks = []
            lines = out.strip().splitlines()
            for line in lines[1:]:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 3:
                    ssid = parts[0] or '<hidden>'
                    ch = int(parts[1])
                    sig = float(parts[2])
                    freq = channel_to_freq_mhz(ch)
                    dbm = (sig / 100.0) * 50.0 - 100.0
                    networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
            return networks
    except Exception:
        return []
    return []

# ------------------ Live plotting data structures ------------------
MAX_POINTS = 60
UPDATE_INTERVAL_MS = 2000

history_rssi = defaultdict(lambda: deque(maxlen=MAX_POINTS))
history_snr = defaultdict(lambda: deque(maxlen=MAX_POINTS))
history_srri = defaultdict(lambda: deque(maxlen=MAX_POINTS))
time_axis = deque(maxlen=MAX_POINTS)
_lock = threading.Lock()

def dbm_to_snr(dbm):
    """Arvioi SNR signaalitasosta (dBm) olettaen kohinatasoksi −100 dBm."""
    if dbm is None:
        return 0.0
    return max(0.0, dbm + 100.0)

def snr_to_srri(snr):
    """Normalisoi SNR arvoksi 0–100."""
    return max(0.0, min(100.0, (snr / 70.0) * 100.0))

def scanner_thread(stop_event):
    while not stop_event.is_set():
        nets = scan_wifi()
        t = time.time()
        with _lock:
            time_axis.append(t)
            seen = set()
            for net in nets:
                ssid = net.get('ssid') or '<hidden>'
                dbm = net.get('dbm', -100.0)
                snr = dbm_to_snr(dbm)
                srri = snr_to_srri(snr)
                history_rssi[ssid].append(dbm)
                history_snr[ssid].append(snr)
                history_srri[ssid].append(srri)
                seen.add(ssid)
            for s in list(history_rssi.keys()):
                if s not in seen:
                    history_rssi[s].append(np.nan)
                    history_snr[s].append(np.nan)
                    history_srri[s].append(np.nan)
        time.sleep(UPDATE_INTERVAL_MS / 1000.0)

# ------------------ Plotting ------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
plt.subplots_adjust(hspace=0.35)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def update_plot(frame):
    with _lock:
        if not time_axis:
            return
        times = np.array(time_axis)
        times_rel = times - times[-1]

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.set_title("📡 RSSI (Received Signal Strength Indicator)")
        ax1.set_xlabel("Aika (s)")
        ax1.set_ylabel("Signaali (dBm)")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-110, -30)

        ax2.set_title("🔉 SNR (Signal-to-Noise Ratio)")
        ax2.set_xlabel("Aika (s)")
        ax2.set_ylabel("SNR (dB)")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 70)

        ax3.set_title("📊 SRRI (Signal Received Ratio Indicator)")
        ax3.set_xlabel("Aika (s)")
        ax3.set_ylabel("SRRI (0–100)")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        ssids = sorted(history_rssi.keys(), key=lambda s: np.nanmax(np.array(history_rssi[s])) if history_rssi[s] else -999)
        top_ssids = ssids[:5]  # näytetään 5 vahvinta

        for i, ssid in enumerate(top_ssids):
            color = colors[i % len(colors)]
            rssi = np.array(history_rssi[ssid])
            snr = np.array(history_snr[ssid])
            srri = np.array(history_srri[ssid])

            ax1.plot(times_rel, rssi, label=f"{ssid}", color=color, linewidth=1.3)
            ax2.plot(times_rel, snr, label=f"{ssid}", color=color, linewidth=1.3)
            ax3.plot(times_rel, srri, label=f"{ssid}", color=color, linewidth=1.3)

        for ax in (ax1, ax2, ax3):
            ax.legend(loc="upper left", fontsize="small")

def main():
    stop_event = threading.Event()
    t = threading.Thread(target=scanner_thread, args=(stop_event,), daemon=True)
    t.start()
    ani = animation.FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t.join(timeout=2)

if __name__ == "__main__":
    main()
