#!/bin/bash
# parallel ranged download of valid_videos.zip, 16 parts, resumable
U=https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip
T=75375143328; N=16; P=$(( (T + N - 1) / N ))
cd ~/work/iclr/ptest/dl/parts
for i in $(seq 0 $((N-1))); do
  s=$((i*P)); e=$(( (i+1)*P - 1 )); [ $e -ge $T ] && e=$((T-1))
  want=$((e-s+1))
  ( for try in 1 2 3 4 5 6 7 8 9 10; do
      have=$(stat -c %s p$i 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      curl -s --retry 5 -r $((s+have))-$e $U >> p$i
    done; echo "part $i done $(stat -c %s p$i)/$want" ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo p$i; done) > ../valid_videos.zip && echo CONCAT_OK $(stat -c %s ../valid_videos.zip)
