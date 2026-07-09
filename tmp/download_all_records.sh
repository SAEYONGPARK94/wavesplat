#!/bin/bash
# Download all remaining Guimaraes/WASS processed nc records into the repo.
# Prefer *_short.nc (analysis-sufficient); skip already-present files.
set -u
BASE="https://data-dataref.ifremer.fr/stereo"
DEST="/home/vexit/wavesplat/wavesplat/data/real_bergamasco"
SITES="AA_2014 AA_2015 BS_2011 BS_2013 LJ_2018 YS_2017"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

for site in $SITES; do
  recs=$(curl -s -m 30 "$BASE/$site/" | grep -oiE 'href="[0-9][^"]+/"' | sed 's/href="//;s/"//')
  for rec in $recs; do
    ncurl="$BASE/$site/${rec}nc/"
    files=$(curl -s -m 30 "$ncurl" | grep -oiE 'href="[^"]+\.nc"' | sed 's/href="//;s/"//')
    [ -z "$files" ] && { log "SKIP $site/$rec (no nc)"; continue; }
    # prefer *_short.nc; if none, take the plain full .nc
    shorts=$(echo "$files" | grep -i '_short.nc' || true)
    picks=${shorts:-$files}
    mkdir -p "$DEST/$site"
    for fnc in $picks; do
      out="$DEST/$site/$fnc"
      if [ -f "$out" ]; then
        # verify size matches remote (resume/skip)
        rsz=$(curl -s -I -m 20 "${ncurl}${fnc}" | grep -i content-length | tr -d '\r' | awk '{print $2}')
        lsz=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ "${rsz:-0}" = "$lsz" ]; then log "HAVE $site/$fnc ($((lsz/1024/1024))MB)"; continue; fi
        log "RESUME $site/$fnc (local $((lsz/1024/1024))MB / remote $((${rsz:-0}/1024/1024))MB)"
      else
        log "GET $site/$fnc"
      fi
      curl -sS -C - -o "$out" "${ncurl}${fnc}" && log "DONE $site/$fnc ($(($(stat -c%s "$out")/1024/1024))MB)" || log "FAIL $site/$fnc"
    done
  done
done
log "ALL DONE. total: $(du -sh "$DEST" 2>/dev/null | awk '{print $1}')"
