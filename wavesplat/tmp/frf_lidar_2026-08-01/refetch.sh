#!/bin/bash
cd "$(dirname "$0")/ws"
while read -r u; do
  f=$(basename "$u")
  exp=$(curl -sIL --max-time 60 "$u" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  act=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ -z "$exp" ] || [ "$act" != "$exp" ]; then
    curl -sfL --max-time 1800 -o "$f" "$u" && echo "OK $f"
  fi
done < ../dl_list.txt
echo ALLDONE > ../dl.done
