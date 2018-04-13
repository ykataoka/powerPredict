#! /bin/bash
for f in *.png; do
convert $f eps2:${f%.png}.eps;
done
