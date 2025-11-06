#!/usr/bin/env python3
import sys
import re
import subprocess
import platform
from collections import defaultdict
import math

import matplotlib.pyplot as plt
import numpy as np

# ---------- Utility: channel <-> frequency helpers ----------
# 2.4 GHz channels 1-14
def channel_to_freq_mhz(channel):
    # channel may be int or string like "36"
    try:
        ch = int(channel)
    except Exception:
        return None
    # 2.4 GHz
    if 1 <= ch <= 14:
        # Channel 1 -> 2412 MHz, channel n -> 2407 + 5*n
        return 2407 + 5 * ch
    # 5 GHz common channels (approx)
    # formula not continuous; use table for common channels:
    table = {
        36: 5180, 40: 5200, 44: 5220, 48: 5240,
        52: 5260, 56:5280, 60:5300, 64:5320,
        100:5500,104:5520,108:5540,112:5560,116:5580,120:5600,124:5620,128:5640,
        132:5660,136:5680,140:5700,144:5720,
        149:5745,153:5765,157:5785,161:5805,165:5825
    }
    return table.get(ch)

def freq_string_to_mhz(s):
    """Parse strings like '2.462 GHz' or '2462 MHz' or 'Channel 6'."""
    if s is None:
        return None
    s = s.strip()
    # MHz
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*MHz', s, re.I)
    if m:
        return float(m.group(1))
    # GHz
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*GHz', s, re.I)
    if m:
        return float(m.group(1)) * 1000.0
    # channel
    m = re.search(r'channel[:=\s]*([0-9]+)', s, re.I)
    if m:
        return channel_to_freq_mhz(m.group(1))
    # lone number, could be channel or mhz
    m = re.search(r'\b([0-9]{2,4})\b', s)
    if m:
        val = int(m.group(1))
        if val < 300:  # treat as channel
            return channel_to_freq_mhz(val)
        else:
            return float(val)  # assume MHz
    return None

# ---------- Parsers for platform-specific scan outputs ----------
def parse_iwlist(output):
    # Linux: iwlist scan
    # look for Frequency:X.XXX GHz and ESSID and Quality
    networks = []
    cells = re.split(r'Cell \d+ - ', output)
    for c in cells:
        if not c.strip():
            continue
        ssid_m = re.search(r'ESSID:"([^"]*)"', c)
        freq_m = re.search(r'Frequency:([0-9.]+)\s*GHz', c)
        qual_m = re.search(r'Quality[=\:]\s*([0-9/]+)', c) or re.search(r'Signal level[=\:]\s*([-\d]+) dBm', c)
        ch_m = re.search(r'Channel[:=]?\s*([0-9]+)', c)
        ssid = ssid_m.group(1) if ssid_m else None
        freq = None
        if freq_m:
            freq = float(freq_m.group(1)) * 1000.0
        elif ch_m:
            freq = channel_to_freq_mhz(int(ch_m.group(1)))
        elif ssid:
            # fallback: search any MHz/MHz-like token
            f = re.search(r'([0-9]{3,4})\s*MHz', c)
            if f:
                freq = float(f.group(1))
        # parse dBm if present
        dbm = None
        dbm_m = re.search(r'Signal level[=\:]\s*([-\d]+)\s*dBm', c)
        if dbm_m:
            dbm = float(dbm_m.group(1))
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

def parse_nmcli(output):
    # nmcli -f SSID,CHAN,SIGNAL dev wifi
    networks = []
    lines = output.strip().splitlines()
    # skip header if present
    for line in lines:
        if not line.strip():
            continue
        # naive split: last columns are CHAN and SIGNAL
        parts = [p.strip() for p in re.split(r'\s{2,}', line)]
        # Try to detect if header exists
        if parts[0].lower().startswith('ssid') or parts[0].lower().startswith('in-use'):
            continue
        ssid = parts[0]
        chan = None
        sig = None
        if len(parts) >= 2:
            # find channel token anywhere
            for token in parts[1:]:
                if re.match(r'^\d+$', token):
                    chan = token
                    break
        if len(parts) >= 3:
            # try last as signal %
            last = parts[-1]
            m = re.search(r'(\d+)', last)
            if m:
                sig = float(m.group(1))
        freq = channel_to_freq_mhz(int(chan)) if chan and chan.isdigit() else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': None if sig is None else (sig-100)})  # approximate dBm
    return networks

def parse_netsh(output):
    # Windows: netsh wlan show networks mode=bssid
    networks = []
    # sections separated by "SSID X : name"
    ssid_blocks = re.split(r'\r?\n\s*SSID\s+\d+\s*:\s*', output)
    # first block is header
    for blk in ssid_blocks[1:]:
        lines = blk.splitlines()
        ssid = lines[0].strip()
        freq = None
        dbm = None
        channel = None
        # search in block
        for L in lines[1:]:
            mchan = re.search(r'Channel\s*:\s*(\d+)', L)
            if mchan:
                channel = int(mchan.group(1))
            msignal = re.search(r'Signal\s*:\s*([0-9]+)%', L)
            if msignal:
                # convert % to approximate dBm (very rough)
                pct = int(msignal.group(1))
                dbm = (pct / 2) - 100  # rough map 0%->-100,100%->-50
        freq = channel_to_freq_mhz(channel) if channel else None
        networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': dbm})
    return networks

def parse_airport(output):
    # macOS: /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s
    networks = []
    lines = output.splitlines()
    if not lines:
        return networks
    # header -> columns: SSID BSSID RSSI CHANNEL HT CC SECURITY (SSID may have spaces)
    # We'll parse based on columns start positions from header
    header = lines[0]
    # Try column splitting by multiple spaces:
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) >= 3:
            ssid = parts[0]
            rssi = None
            ch = None
            # find channel token like "11" or "6,1"
            for token in parts:
                if re.match(r'^\d{1,3}(?:,.*)?$', token):
                    # token could be "11" or "36" or "36,1"
                    chtok = token.split(',')[0]
                    if chtok.isdigit():
                        ch = int(chtok)
                        break
            # RSSI likely available as negative dBm in parts
            for token in parts:
                if re.match(r'^-?\d+\s*$', token):
                    val = int(token.strip())
                    # RSSI typical range -100..0
                    if -120 < val < 0:
                        rssi = val
                        break
            freq = channel_to_freq_mhz(ch) if ch else None
            networks.append({'ssid': ssid, 'freq_mhz': freq, 'dbm': rssi})
    return networks

# ---------- Platform scan functions ----------
def scan_linux():
    # Try nmcli first (more common on modern distros)
    try:
        out = subprocess.check_output(['nmcli', '-f', 'SSID,CHAN,SIGNAL', 'device', 'wifi', 'list'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True)
        parsed = parse_nmcli(out)
        if parsed:
            return parsed
    except Exception:
        pass
    # fallback to iwlist
    try:
        # find wireless interface
        iwconfig_out = subprocess.check_output(['iwconfig'], stderr=subprocess.DEVNULL, universal_newlines=True)
        m = re.search(r'([a-zA-Z0-9]+)\s+IEEE 802.11', iwconfig_out)
        iface = m.group(1) if m else 'wlan0'
        out = subprocess.check_output(['sudo', 'iwlist', iface, 'scan'], stderr=subprocess.DEVNULL, universal_newlines=True)
        return parse_iwlist(out)
    except Exception:
        return []

def scan_windows():
    try:
        out = subprocess.check_output(['netsh', 'wlan', 'show', 'networks', 'mode=bssid'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True)
        return parse_netsh(out)
    except Exception:
        return []

def scan_macos():
    try:
        out = subprocess.check_output(['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s'],
                                      stderr=subprocess.DEVNULL, universal_newlines=True)
        return parse_airport(out)
    except Exception:
        # try /usr/sbin/airport
        try:
            out = subprocess.check_output(['/usr/sbin/airport', '-s'], stderr=subprocess.DEVNULL, universal_newlines=True)
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

# ---------- Main: scan and plot ----------
def plot_networks(networks):
    # Filter networks that have a frequency
    nets_with_freq = [n for n in networks if n.get('freq_mhz') is not None]
    if not nets_with_freq:
        print("Ei löydetty verkkoja tai taajuuksia. Varmista, että laite tukee skannausta ja että skripti suoritetaan tarvittavilla oikeuksilla.")
        return

    freqs = [n['freq_mhz'] for n in nets_with_freq]
    ssids = [n.get('ssid') or '<unknown>' for n in nets_with_freq]
    dbms = [n.get('dbm') if n.get('dbm') is not None else -100 for n in nets_with_freq]

    # histogram of frequencies (bin width 5/20/??)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    # convert to GHz for nicer ticks
    freqs_ghz = [f/1000.0 for f in freqs]
    bins = np.linspace(min(freqs_ghz)-0.05, max(freqs_ghz)+0.05, 20)
    plt.hist(freqs_ghz, bins=bins)
    plt.xlabel('Taajuus (GHz)')
    plt.ylabel('Verkkojen lukumäärä')
    plt.title('Wi-Fi verkkojen taajuushistogrammi')

    # scatter: frequency vs signal (dbm)
    plt.subplot(2,1,2)
    plt.scatter(freqs_ghz, dbms)
    for x,y,label in zip(freqs_ghz, dbms, ssids):
        plt.text(x, y+1.5, label, fontsize=8, rotation=25, alpha=0.7)
    plt.xlabel('Taajuus (GHz)')
    plt.ylabel('Signal strength (approx dBm)')
    plt.title('Taajuus vs signaalin voimakkuus (jokainen verkko)')
    plt.tight_layout()
    plt.show()

def main():
    print("Skannataan Wi-Fi-verkkoja järjestelmältä...")
    nets = scan_wifi()
    if not nets:
        print("Ei löytynyt verkkoja tai skannaukseen ei ollut pääsyä.")
        return
    print(f"Löytyi {len(nets)} verkkoa (sis. mahdolliset joissa ei taajuutta):")
    for n in nets:
        print(f"  SSID: {n.get('ssid')}, freq: {n.get('freq_mhz')}, dbm: {n.get('dbm')}")
    plot_networks(nets)

if __name__ == "__main__":
    main()
