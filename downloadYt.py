#!/usr/bin/env python

import yt_dlp
import os
import gc
import random
import time
import subprocess

import random
import requests
import subprocess
import time

import tqdm
import argparse
from functools import wraps
import numpy as np

os.environ["PATH"] = "/work/users/s/m/smerrill/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]

def load_proxies(file_path='proxies.txt'):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def is_proxy_alive(proxy):
    try:
        response = requests.get(
            'https://httpbin.org/ip',
            proxies={'http': proxy, 'https': proxy},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ Proxy works: {response.json()['origin']}")
            return True
    except:
        pass
    print(f"❌ Dead proxy: {proxy}")
    return False
def retry(times=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Attempt {i+1} failed: {e}")
                    time.sleep(delay)
            print("❌ All attempts failed.")
            return []
        return wrapper
    return decorator


@retry(times=3, delay=2)
def fetch_proxies_proxyscrape():
    print("🌐 Fetching US HTTP & SOCKS proxies from ProxyScrape...")
    proxies = []

    base_url = "https://api.proxyscrape.com/v2/?request=getproxies&timeout=3000&country=us&ssl=all&anonymity=all"

    types = {
        "http": f"{base_url}&protocol=http",
        "socks4": f"{base_url}&protocol=socks4",
        "socks5": f"{base_url}&protocol=socks5"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for proto, url in types.items():
        try:
            time.sleep(1.5)  # Throttle to avoid 429
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                print(f"🚫 ProxyScrape ({proto}) says: Too Many Requests. Skipping...")
                continue
            response.raise_for_status()
            new_proxies = [f"{proto}://{line.strip()}" for line in response.text.splitlines() if line.strip()]
            proxies.extend(new_proxies)
        except Exception as e:
            print(f"❌ ProxyScrape ({proto}) failed: {e}")
    
    return proxies

@retry(times=3, delay=2)
def fetch_proxies_geonode():
    print("🌐 Fetching US HTTP proxies from GeoNode...")
    url = "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&country=US&protocols=http"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        proxies = [f"http://{proxy['ip']}:{proxy['port']}" for proxy in data['data']]
        return proxies
    except Exception as e:
        print(f"❌ GeoNode failed: {e}")
        return []
    
@retry(times=3, delay=2)
def fetch_proxies_proxylist_download():
    print("🌐 Fetching US HTTP proxies from proxy-list.download...")
    url = "https://www.proxy-list.download/api/v1/get?type=http&country=US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        proxies = [f"http://{line.strip()}" for line in response.text.splitlines() if line.strip()]
        return proxies
    except Exception as e:
        print(f"❌ proxy-list.download failed: {e}")
        return []

def fetch_all_us_proxies():
    proxies = set()
    sources = [
        fetch_proxies_proxyscrape,
        fetch_proxies_geonode,
        fetch_proxies_proxylist_download
    ]
    for fetch_func in sources:
        proxies.update(fetch_func())

    # Final filter: Only include HTTP and SOCKS proxies
    valid_prefixes = ("http://", "socks4://", "socks5://")
    filtered = [p for p in proxies if p.startswith(valid_prefixes)]

    print(f"✅ Total valid HTTP/SOCKS US proxies fetched: {len(filtered)}")
    return filtered


def is_youtube_accessible(proxy):
    test_url = "https://www.youtube.com"
    try:
        resp = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=5)
        return resp.status_code == 200
    except:
        return False


def download_full_youtube_video(video_id: str, proxy: str,  output_dir: str = ".") -> str:
    """
    Downloads the full YouTube video and saves it as MP4.

    Requires: yt-dlp and ffmpeg (for merging video/audio)
    Returns: Full path to downloaded .mp4 file
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    download_command = [
        "yt-dlp",
        "--proxy", proxy,
        "-f", "bestvideo+bestaudio/best",
        "-o", output_path,
        "--merge-output-format", "mp4",
        url
    ]

    print(f"⬇️ Downloading full video from {url}...")
    subprocess.run(download_command, check=True)
    print(f"✅ Video saved to: {output_path}")

    return output_path


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')

  #parser.add_argument("-s", "--start_index", type=int, default=0, help='YoutubeID Index to start on in YoutubeID File')
  #parser.add_argument("-e", "--end_index", type=int, default=100, help='YoutubeID Index to end on in YoutubeID File')
  parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/Albemarle', help='Path to Save YT vids to')
  #parser.add_argument("-vp", "--vid_path", type=str, default='/work/users/s/m/smerrill/LocalView/vids.npy', help='Path to YoutubeID file.  This will also be where output featuers are saved')
  args = vars(parser.parse_args())

 
  #os.makedirs(args['save_path'] + '/audio', exist_ok=True)

  # get downloaded vids
  #downloaded_vids = os.listdir(args['save_path'] + '/audio')
  #downloaded_vids = [x.split('.')[0] for x in downloaded_vids]

  # Here are the youtube ids used by original VM-NET
  #vid_ids = np.load(args['vid_path'], allow_pickle=True)

  #vid_ids = vid_ids[args['start_index']:args['end_index']]
  vid_ids = ["_91XXbXeQD4",
    "xfVck0_Q84w",
    "o3f7y_9mHcE",
    "3BtZN2Tye08",
    "Bcl4e29n7m4",
    "cF4uYrMPQ24",
    "5YMkxWBgdtY",
    "8TdTe--0CUs",
    "wyl3i48JFkA",
    "82YE6lBeZA8",
    "GarPnbypXRk",
    "BQYLZNhIEDE",
    "LUO7q6gjpvk",
    "5GglIIs8-B8",
    "wbgxA2KNbiw",
    "xJvnLptI_SU"]

  all_proxies = []
  i = 0
  for vid in tqdm.tqdm(vid_ids):
      # video id      
      #if vid in downloaded_vids:
      #  print(f"VID: {vid} alread processed, skipping")
      #  continue


      print(f"processing VID: {vid}")
      if i >= len(all_proxies):
        i = 0
        all_proxies = fetch_all_us_proxies()

      random.shuffle(all_proxies)

      success = False

      while i < len(all_proxies):
        proxy = all_proxies[i]
        if not proxy.startswith(("http://", "socks4://", "socks5://")):
            continue

        if is_youtube_accessible(proxy):
            print("Attempting Downlaod")
            success = download_full_youtube_video(vid, proxy, output_dir=args['save_path'])
            if success:
                print("✅ SUCCESS — Video downloaded.")
                break
        else:
            print(f"⚠️ Proxy not allowed by YouTube: {proxy}")

        i += 1

      if not success:
          print("💀 All proxies failed or timed out. Continuing to next video...")
      
  print("Download Complete")
